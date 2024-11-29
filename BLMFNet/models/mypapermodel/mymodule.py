import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, Parameter, Linear, Sigmoid, Softmax
import numpy as np


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class BSM(nn.Module):
    def __init__(self, all_channel=64):
        super(BSM, self).__init__()
        self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = SpatialAttention()
        self.dconv1 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=3, padding=3)
        self.dconv3 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=5, padding=5)
        self.dconv4 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=7, padding=7)
        self.fuse_dconv = nn.Conv2d(all_channel, 64, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        x = x * y
        x1 = self.dconv1(x)
        x2 = self.dconv2(x)
        x3 = self.dconv3(x)
        x4 = self.dconv4(x)
        out = self.fuse_dconv(torch.cat((x1, x2, x3, x4), dim=1))
        edge_pred = self.pred(out)
        return out, edge_pred
        # return out

class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)  # 256 64 64
        xo = 2 * x * wei
        x1 = 2 * residual * (1 - wei)
        return xo, x1


if __name__ == "__main__":
    model = BSM()
    model.train()
    sar = torch.randn(2, 64, 256, 256)
    opt = torch.randn(2, 64, 256, 256)
    print(model)
    print("input:", opt.shape)
    output = model(opt)
    out = output[0]
    edge = output[1]
    print("output:", out.shape)
    print("output:", edge.shape)

    # model = CAM_Module(in_dim=256).to(device='cuda:0')
    # input_t = torch.randn((2, 256, 64, 64), device='cuda:0')
    # out = model(input_t)
    # print(out.shape)

    # model = AFF(channels=256).to(device='cuda:0')
    # input_opt = torch.randn((2, 256, 64, 64), device='cuda:0')
    # input_sar = torch.randn((2, 256, 64, 64), device='cuda:0')
    # out1, out2 = model(input_opt,input_sar)
    # print(out1.shape, out2.shape)
