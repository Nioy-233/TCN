"""  
用法示例 Example Usage  
--------------------------  

以下示例展示如何使用TemporalConvNet对一组(batch, 特征数, 序列长度)格式的一维序列数据进行处理。  

>>> import torch  
>>> # 假设输入为 batch_size=4, 特征数=8, 序列长度=30  
>>> x = torch.randn(4, 8, 30)  

>>> # 初始化TCN，2层，通道数分别为16和32，卷积核大小为3  
>>> model = TemporalConvNet(num_inputs=8, num_channels=[16, 32], kernel_size=3, dropout=0.1)  

>>> # 模型前向传播  
>>> y = model(x)  # 输出形状为 [4, 32, 30]  

>>> print(y.shape)  
torch.Size([4, 32, 30])  

# 如果你的数据输入格式是 [batch_size, 时间步, 特征数]，记得先转换  
>>> x_raw = torch.randn(4, 30, 8)  
>>> x = x_raw.permute(0, 2, 1)  # 转为 [4, 8, 30]  
>>> y = model(x)  

# 可以自定义输出层（如分类或回归）  
>>> import torch.nn as nn  
>>> class MyTCNClassifier(nn.Module):  
...     def __init__(self, input_dim, tcn_channels, num_classes):  
...         super().__init__()  
...         self.tcn = TemporalConvNet(input_dim, tcn_channels)  
...         self.fc = nn.Linear(tcn_channels[-1], num_classes)  
...     def forward(self, x):  
...         out = self.tcn(x)      # [batch, channels, seq_len]  
...         out = out[:, :, -1]   # 取最后时刻  
...         return self.fc(out)  
>>> clf = MyTCNClassifier(8, [16,32], 10)  
>>> y_pred = clf(x)     # y_pred.shape == [4, 10]  

"""  

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """  
    移除输入张量的最后部分，确保张量在时间维度上的长度适合后续计算。  

    参数:  
    - chomp_size: 需要移除的时间步数，影响输出的时间维度。  

    输入是一个三维张量 [batch_size, channels, length]。  
    最终输出是一个三维张量 [batch_size, channels, length - chomp_size]。  
    """  
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """  
    构建一个基础的时间卷积块，包括卷积、激活、丢弃和残差连接。  

    参数:  
    - n_inputs: 输入通道数。  
    - n_outputs: 输出通道数。  
    - kernel_size: 卷积核大小。  
    - stride: 卷积步长。  
    - dilation: 卷积膨胀系数。  
    - padding: 填充大小。  
    - dropout: 丢弃率，默认值为0.2。  

    输入是一个三维张量 [batch_size, n_inputs, length]。  
    最终输出是一个三维张量 [batch_size, n_outputs, length]。  
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """  
    构建一个由多个 TemporalBlock 组成的时间卷积网络。  

    参数:  
    - num_inputs: 输入通道数。  
    - num_channels: 每个层的输出通道数列表。  
    - kernel_size: 卷积核大小，默认值为2。  
    - dropout: 丢弃率，默认值为0.2。  

    输入是一个三维张量 [batch_size, num_inputs, length]。  
    最终输出是一个三维张量 [batch_size, num_channels[-1], length]。  
    """  
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
