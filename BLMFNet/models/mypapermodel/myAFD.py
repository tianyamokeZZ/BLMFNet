import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import ConvModule
from torch import einsum


class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAtt, self).__init__()
        # 1x1 卷积层
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # 归一化层
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Forward function."""
        # 计算平均池化的注意力图
        atten = torch.mean(x, dim=(2, 3), keepdim=True)
        # 应用1x1卷积和归一化
        atten = self.conv_1x1(atten)
        atten = self.bn(atten)
        return x, atten


class AFD(nn.Module):
    "Active fusion decoder"

    def __init__(self, s_channels, c_channels, h=8):
        super(AFD, self).__init__()
        self.s_channels = s_channels
        self.c_channels = c_channels
        self.h = h
        self.scale = h ** - 0.5
        self.spatial_att = ChannelAtt(s_channels, s_channels)
        self.context_att = ChannelAtt(c_channels, c_channels)
        self.qkv = nn.Linear(s_channels + c_channels, (s_channels + c_channels) * 3, bias=False)
        self.proj = nn.Linear(s_channels + c_channels, s_channels + c_channels)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, sp_feat, co_feat):
        # **_att: B x C x 1 x 1
        s_feat, s_att = self.spatial_att(sp_feat)
        c_feat, c_att = self.context_att(co_feat)
        b = s_att.shape[0]  # h = 1, w = 1
        sc_att = torch.cat([s_att, c_att], 1).view(b, -1)  # [B,2C]
        qkv = self.qkv(sc_att).reshape(b, 1, 3, self.h, (self.c_channels + self.s_channels) // self.h).permute(2, 0, 3,
                                                                                                               1,
                                                                                                               4)  # [B,2C] -> [B,6C] -> [B,1,3,h,2C // h] -> [3,B,h,1,2C // h]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,h,1,2C // h]
        k_softmax = k.softmax(dim=1)  # channel-wise softmax operation
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)  # [B,h,2C // h ,2C // h]
        fuse_weight = self.scale * einsum("b h n k, b h k v -> b h n v", q,
                                          k_softmax_T_dot_v)  # [B,h,1,2C // h]
        fuse_weight = fuse_weight.transpose(1, 2).reshape(b, -1)  # [B,C]
        fuse_weight = self.proj(fuse_weight)
        fuse_weight = self.proj_drop(fuse_weight)
        fuse_weight = fuse_weight.reshape(b, -1, 1, 1)  # [B,C,1,1]
        fuse_s, fuse_c = fuse_weight[:, :self.s_channels], fuse_weight[:, -self.c_channels:]
        out = (1 + fuse_s) * s_feat + (1 + fuse_c) * c_feat
        return out


if __name__ == "__main__":
    model = AFD(s_channels=256, c_channels=256)
    model.train()
    seg = torch.randn(2, 256, 256, 256)
    boun = torch.randn(2, 256, 256, 256)
    # print(model)
    output = model(seg, boun)
    print("input:", seg.shape, boun.shape)
    print("output0:", output.shape)
    # print("output1:", output[1].shape)
    # print("output2:", output[2].shape)

