#!/usr/bin/python3
# coding=utf-8

#  改了整合模块

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):  # 参数初始化
    for n, m in module.named_children():  # 返回的是迭代器iterator
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):  # !!!!!新加的
            weight_init(m)
        elif isinstance(m, nn.ReLU):  # ！！！！！新加的
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual, inplace=True)


class ResNet(nn.Module):  # 特征提取网络
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * 4))

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('resnet50-19c8e357.pth'), strict=False)


"""这一部分是邻接互补模块的代码"""


class CFM(nn.Module):
    def __init__(self):  # 可以学CA在这里设定输入输出的维数，def __init__(self, h_planes, l_planes):
        super(CFM, self).__init__()  # 先改为256
        self.conv1h = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn1h = nn.BatchNorm2d(128)
        # self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3h = nn.BatchNorm2d(128)
        # self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn1v = nn.BatchNorm2d(128)
        # self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3v = nn.BatchNorm2d(128)
        # self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn4v   = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:  # 这里参照的F3Net的代码写的，后面看要不要该改成门控单元那样的，输入输出的通道数关系要注意
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')  # 对高级特征进行上采样，使得和输入的低级特征尺寸大小一样。
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        # out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        # out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse = out1h * out1v
        out3h = fuse + out1h
        # out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = fuse + out1v
        # out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        outF = torch.cat((out3h, out3v), 1)
        return outF

    def initialize(self):
        weight_init(self)

class CFM1(nn.Module):
    def __init__(self):  # 可以学CA在这里设定输入输出的维数，def __init__(self, h_planes, l_planes):
        super(CFM1, self).__init__()  # 先改为256
        self.conv1h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1h = nn.BatchNorm2d(256)
        # self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3h = nn.BatchNorm2d(256)
        # self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1v = nn.BatchNorm2d(256)
        # self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3v = nn.BatchNorm2d(256)
        # self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn4v   = nn.BatchNorm2d(64)
        self.sa = SpatialAttentionModule()
        self.squeeze1 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:  # 这里参照的F3Net的代码写的，后面看要不要该改成门控单元那样的，输入输出的通道数关系要注意
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')  # 对高级特征进行上采样，使得和输入的低级特征尺寸大小一样。
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        # out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        # out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse = out1h * out1v
        out3h = fuse + out1h
        # out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = fuse + out1v
        # out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        outF = torch.cat((out3h, out3v), 1)
        outS = torch.cat((left, down), 1)
        outSA = self.sa(outS)
        outFF = outF+ outSA

        return self.squeeze1(outFF)

    def initialize(self):
        weight_init(self)


class CPFE(nn.Module):
    def __init__(self, feature_layer=None, out_channels=32):
        super(CPFE, self).__init__()

        self.dil_rates = [3, 5, 7]

        # Determine number of in_channels from VGG-16 feature layer
        if feature_layer == 'conv5':
            self.in_channels = 2048
        elif feature_layer == 'conv4':
            self.in_channels = 1024
        elif feature_layer == 'conv3':
            self.in_channels = 512
        elif feature_layer == 'conv2':
            self.in_channels = 256
        elif feature_layer == 'conv1':
            self.in_channels = 64

        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False)  #
        self.conv_dil_3 = nn.Conv2d(in_channels=(self.in_channels+32), out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=(self.in_channels+32), out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=(self.in_channels+32), out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)
        self.conv_input = nn.Conv2d(self.in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.ca = CBAM(128)
        # self.ca = ChannelAttention(in_channel)
        self.bn = nn.BatchNorm2d(256)  # !!!!!!!
        self.squeeze = nn.Sequential(nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        c1 =  torch.cat((input_, conv_1_1_feats), 1)
        conv_dil_3_feats = self.conv_dil_3(c1)
        c2 = torch.cat((input_, conv_dil_3_feats), 1)
        conv_dil_5_feats = self.conv_dil_5(c2)
        c3= torch.cat((input_, conv_dil_5_feats), 1)
        conv_dil_7_feats = self.conv_dil_7(c3)

        new_input = self.conv_input(input_)
        conv_ca = self.ca(new_input)  # 128
        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats, conv_ca),
                                 dim=1)  # 256
        bn_feats = F.relu(self.bn(concat_feats))  # 256
        bn_feat = self.squeeze(bn_feats)  # 128
        return bn_feat

    def initialize(self):
        weight_init(self)

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def initialize(self):
        weight_init(self)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    def initialize(self):
        weight_init(self)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

    def initialize(self):
        weight_init(self)

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out

    def initialize(self):
        weight_init(self)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = torch.sigmoid(self.conv2d(out))
        return out * x

    def initialize(self):
        weight_init(self)


class CA(nn.Module):  # 将注意力模块的输出设定为128
    def __init__(self, in_channel_left, in_channel_down):
        super(CA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 128, kernel_size=1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(in_channel_down, 128, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 128
        down = down.mean(dim=(2, 3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)  # 128
        down = torch.sigmoid(self.conv2(down))
        return left * down

    def initialize(self):
        weight_init(self)


""" Self Refinement Module """


class SRM(nn.Module):
    def __init__(self, in_channel):
        super(SRM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 256
        out2 = self.conv2(out1)  # 512个通道
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)

    def initialize(self):
        weight_init(self)


""" Feature Interweaved Aggregation Module """


class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(FAM, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(256)

        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')  # 完成上采样
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_1 * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

    def initialize(self):
        weight_init(self)


class SA(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(SA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_down, 512, kernel_size=3, stride=1, padding=1)
        self.squeeze = nn.Sequential(nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down_1 = self.conv2(down)  # wb
        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')
        w, b = down_1[:, :256, :, :], down_1[:, 256:, :, :]
        s = F.relu(w * left + b, inplace=True)
        return self.squeeze(s)

    def initialize(self):
        weight_init(self)


class GCPANet(nn.Module):
    def __init__(self, cfg):
        super(GCPANet, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet()

        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=1,
                               padding=1)  # 这个卷积输出通道上的选取用的是256，是不是应该先用一个大的通道数了再换成小的。
        self.bn6 = nn.BatchNorm2d(256)  # 或者说，因为这一路不经过通道上的变化，所以不用管，直接取128

        self.m1 = CPFE(feature_layer='conv1')
        self.m2 = CPFE(feature_layer='conv2')
        self.m3 = CPFE(feature_layer='conv3')
        self.m4 = CPFE(feature_layer='conv4')
        self.m5 = CPFE(feature_layer='conv5')

        self.cfm1 = CFM()
        self.cfm2 = CFM()
        self.cfm3 = CFM()
        self.cfm4 = CFM()
       # self.cfm5 = CFM()
        self.f1 = CFM1()
        self.f2 = CFM1()
        self.f3 = CFM1()
        self.f4 = CFM1()

        self.srm5 = SRM(256)
        self.srm4 = SRM(256)
        self.srm3 = SRM(256)
        self.srm2 = SRM(256)
        self.srm1 = SRM(256)

       # self.linear6 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5_ = self.bkbone(x)
        out6_ = F.relu(self.bn6(self.conv6(out5_)), inplace=True)

        out1_c = self.m1(out1)
        out2_c = self.m2(out2)
       # out1_c = F.relu(self.bn1(self.conv1(out1)), inplace=True)
      # out2_c = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3_c = self.m3(out3)
        out4_c = self.m4(out4)
        out5_c = self.m5(out5_)

       # out1_a = self.sa1(out1_c)
       # out2_a = self.sa2(out2_c)
        # out3_a = self.sa3(out3_c)
        # out4_a = self.sa4(out4_c)
        # out5_a = self.sa5(out5_c)

        out1_b = self.cfm1(out1_c, out2_c)
        out2_b = self.cfm2(out2_c, out3_c)
        out3_b = self.cfm3(out3_c, out4_c)
        out4_b = self.cfm4(out4_c, out5_c)
        #out5_b = self.cfm5(out5_c, out6_)

       # out6 = self.f5(out6_, out5_b)
        out5 = self.f1(out4_b,out6_)
        out4 = self.f2(out3_b,out5 )
        out3 = self.f3(out2_b,out4)
        out2 = self.f4(out1_b,out3 )

        #out5 = self.srm4(out5)
       # out4 = self.srm3(out4)
      #  out3 = self.srm2(out3)
       # out2 = self.srm1(out2)

        # we use bilinear interpolation instead of transpose convolution
        #out6 = F.interpolate(self.linear6(out6), size=x.size()[2:], mode='bilinear')
        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode='bilinear')  # 上采样到输入特征大小
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode='bilinear')
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode='bilinear')
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode='bilinear')
        return out2, out3, out4, out5

    def initialize(self):
        if self.cfg.snapshot:
            try:
                self.load_state_dict(torch.load(self.cfg.snapshot))
            except:
                print("Warning: please check the snapshot file:", self.cfg.snapshot)
                pass
        else:
            weight_init(self)
