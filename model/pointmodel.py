import os
import os.path
import torch
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import netron
import onnx

# torch.set_default_tensor_type(torch.DoubleTensor)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

class STN3d(nn.Module):
    """
    3D空间变换网络，用于学习输入数据的变换。

    参数:
    - channel: 输入数据的通道数。
    """

    def __init__(self, channel):
        super(STN3d, self).__init__()
        # 1D卷积层，用于特征提取
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # 全连接层，用于学习变换矩阵
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        # ReLU激活函数
        self.relu = nn.ReLU()

        # 批归一化层，用于对每层输入进行归一化
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: 3D输入数据，形状为 (batch_size, channel, num_points)。

        返回:
        - x: 变换矩阵，大小为 (batch_size, 3, 3)。
        """
        batchsize = x.size()[0]  # (batch_size, 3, num_points)
        # 特征提取和学习变换矩阵
        x = self.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, num_points)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch_size, 128, num_points)
        x = self.relu(self.bn3(self.conv3(x)))  # (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, 1024, 1)
        x = x.view(-1, 1024)  # (batch_size, 1024)

        x = self.relu(self.bn4(self.fc1(x)))  # (batch_size, 512)
        x = self.relu(self.bn5(self.fc2(x)))  # (batch_size, 256)
        x = self.fc3(x)  # (batch_size, 9)

        # 初始化单位矩阵并将其加到学习到的变换矩阵上
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)  # (batch_size, 9)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden  # (batch_size, 9)
        x = x.view(-1, 3, 3)  # (batch_size, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=8):
        super(PointNetEncoder, self).__init__()
        self.channel = channel
        self.stn = STN3d(channel)  # 空间变换网络，用于对点云进行对齐
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)  # 第一个卷积层
        self.conv2 = torch.nn.Conv1d(64, 128, 1)  # 第二个卷积层
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 第三个卷积层
        self.bn1 = nn.BatchNorm1d(64)  # 第一个批归一化层
        self.bn2 = nn.BatchNorm1d(128)  # 第二个批归一化层
        self.bn3 = nn.BatchNorm1d(1024)  # 第三个批归一化层
        self.global_feat = global_feat  # 是否返回全局特征
        self.feature_transform = feature_transform  # 是否进行特征变换
        self.relu = nn.ReLU()  # ReLU激活函数
        if self.feature_transform:
            self.fstn = STNkd(k=64)  # 特征变换网络
        self.x_norm = nn.LayerNorm(1024)

    def forward(self, x):
        # x = torch.tensor(x, dtype=torch.float32).to(device)
        x = x.transpose(2, 1)
        B, D, N = x.size()  # B: 批次大小, D: 特征维度, N: 点的数量
        # 按照原图进行修改，前三维是位置特征
        # trans = self.stn(x)
        trans = self.stn(x)  # 只对位置特征进行空间变换
        x = x.transpose(2, 1)  # 转置，使点的数量在第二维
        if D > 3:
            feature = x[:, :, 3:]  # 提取额外的特征
            x = x[:, :, :3]  # 只保留位置特征
        # 矩阵乘法，应用空间变换
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)  # 将位置特征和额外特征重新拼接
        x = x.transpose(2, 1)  # 转置回原始形状
        x = self.relu(self.bn1(self.conv1(x)))  # 第一个卷积层，输出维度: [B, 64, N]
        if self.feature_transform:
            trans_feat = self.fstn(x)  # 特征变换
            x = x.transpose(2, 1)  # 转置
            x = torch.bmm(x, trans_feat)  # 应用特征变换
            x = x.transpose(2, 1)  # 转置回原始形状
        else:
            trans_feat = None
        pointfeat = x  # 保存局部特征
        x = self.relu(self.bn2(self.conv2(x)))  # 第二个卷积层，输出维度: [B, 128, N]
        x = self.bn3(self.conv3(x))  # 第三个卷积层，输出维度: [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0]  # 全局池化，输出维度: [B, 1024, 1]
        x = x.view(-1, 1024)  # 展平，输出维度: [B, 1024]
        if self.global_feat:
            return self.x_norm(x), trans, trans_feat  # 返回x,位置特征,全局特征
        return self.x_norm(x), trans, trans_feat
        # else:
        #     x = x.view(-1, 1024, 1).repeat(1, 1, N)  # 重复，输出维度: [B, 1024, N]
        #     return torch.cat([x, pointfeat], 1), trans, trans_feat  # 拼接全局特征和局部特征，输出维度: [B, 1088, N]


class get_model(nn.Module):
    def __init__(self, k=1, normal_channel=True, in_channel=6):
        super(get_model, self).__init__()
        if normal_channel:
            channel = in_channel
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x [batch_size, 1024]
        x, trans, trans_feat = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        # x = F.log_softmax(x, dim=1)
        return x


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = nn.SmoothL1Loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


def test_onnx():
    model = get_model(in_channel=5)
    data = torch.rand([100, 1000, 5])
    data = data.transpose(2, 1)
    # out = model(data)

    torch.onnx.export(model=model, args=data, f='pointmodel.onnx', input_names=['point could'],
                      output_names=['res'])
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load("pointmodel.onnx")), "pointmodel.onnx")
    netron.start("pointmodel.onnx")
    # print("end")

    print("end")


def test_rand():
    model = get_model(in_channel=5)
    data = torch.rand([5, 1000, 5])
    data = data.transpose(2, 1)
    res = model(data)
    print("end")


def test():
    model = get_model(in_channel=8)
    # model.eval()

    # file_paths 是一个包含文件路径的列表
    file_paths = ["../data/R-SIM_RNA/Target_2", "../data/R-SIM_RNA/Target_3", "../data/R-SIM_RNA/Target_4"]
    # 读取所有文件的数据并存储在一个列表中
    data_list = []
    for path in file_paths:
        point = np.loadtxt(path).astype(np.float32)
        data_list.append(point)

    # 将列表转换为一个张量，添加一个额外的维度
    point_data = torch.tensor(data_list, dtype=torch.float32)

    # point_data = point_data.transpose(2, 1)

    import time

    # 记录开始时间
    start_time = time.time()

    # 执行代码
    res = model(point_data)

    # 记录结束时间
    end_time = time.time()

    # 计算执行时间
    execution_time = end_time - start_time
    print(f"代码执行时间: {execution_time:.4f} 秒")
    print(res.shape)
    print("end")


if __name__ == "__main__":
    test()
    # test_rand()
