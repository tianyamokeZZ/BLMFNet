import torch
import torch.nn.functional as F
from torch import nn
from models.mypapermodel.aspp import _ASPP
# from models.utils import initialize_weights
from models.mypapermodel.mymodule import BasicConv2d, AFF, BSM


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if downsample:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class ZZNet(nn.Module):
    def __init__(self, atrous_rates=[6,12,18], channel2=256, channel1=128, channel0=64, num_class=1):
        super(ZZNet, self).__init__()
        self.sar_en1 = _EncoderBlock(1, 64) # 256->128, 1->64
        self.sar_en2 = _EncoderBlock(64, 256)  # 128->64, 64->256
        self.sar_en3 = _EncoderBlock(256, 512)  # 64->32, 256->512
        self.sar_en4 = _EncoderBlock(512, 1024)  # 32->32 *** , 512->1024
        self.sar_en5 = _EncoderBlock(1024, 2048)  # 32->32 *** , 1024->2048

        self.opt_en1 = _EncoderBlock(3, 64, downsample=False) # 256->128, 4->64
        self.opt_en2 = _EncoderBlock(64, 256, downsample=False)  # 128->64, 64->256
        self.opt_en3 = _EncoderBlock(256, 512, downsample=False)  # 64->32, 256->512
        self.opt_en4 = _EncoderBlock(512, 1024, downsample=False)  # 32->32 *** , 512->1024
        self.opt_en5 = _EncoderBlock(1024, 2048)  # 32->32 *** , 1024->2048
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.aspp = nn.Sequential(_ASPP(256 * 2, 256, atrous_rates), nn.Conv2d(256 * 5, 256, kernel_size=1, stride=1, padding=0))
        self.low_level_down = nn.Conv2d(256 * 2, 48, kernel_size=1, stride=1, padding=0)
        self.sar_high_level_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.opt_high_level_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.conv_low_high = nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1)

        self.seg = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        self.locseghead = nn.Sequential(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1))

        self.BSM1 = BSM(all_channel=64)
        self.BSM2 = BSM(all_channel=256)
        self.BSM3 = BSM(all_channel=512)
        self.BSM4 = BSM(all_channel=1024)
        self.bounconv = nn.Conv2d(4, 1, kernel_size=1, padding=0, dilation=1, bias=True)
        initialize_weights(self)

    def forward(self, sar, opt):
        sar_en1 = self.sar_en1(sar)
        sar_en2 = self.sar_en2(sar_en1)
        sar_en3 = self.sar_en3(sar_en2)
        sar_en4 = self.sar_en4(sar_en3)
        sar_en5 = self.sar_en5(sar_en4)

        opt_en1 = self.opt_en1(opt)
        opt_en1_pool = self.MaxPool1(opt_en1)
        opt_en2 = self.opt_en2(opt_en1_pool)
        opt_en2_pool = self.MaxPool2(opt_en2)
        opt_en3 = self.opt_en3(opt_en2_pool)
        opt_en3_pool = self.MaxPool3(opt_en3)
        opt_en4 = self.opt_en4(opt_en3_pool)
        opt_en4_pool = self.MaxPool4(opt_en4)
        opt_en5 = self.opt_en5(opt_en4_pool)

        low_level_features = self.low_level_down(torch.cat([sar_en2, opt_en2_pool], 1))  # 2,512,128,128 ->2,48,128,128
        x0, x1 = self.sar_high_level_down(sar_en5), self.opt_high_level_down(opt_en5)  # 均为2,256,64,64
        high_level_features = torch.cat([x0, x1], 1)
        # loc pred
        loc_seg = self.locseghead(high_level_features)  # 输入 2,512,64,64
        loc_seg = F.interpolate(loc_seg, opt_en1.size()[2:], mode='bilinear')
        high_level_features = self.aspp(high_level_features)  # 2,512,64,64 多尺度特征
        high_level_features = F.interpolate(high_level_features, sar_en2.size()[2:], mode='bilinear')  # 2,256,128,128
        low_high = torch.cat([low_level_features, high_level_features], 1)  # 2,256+48,128,128
        low_high_feature = self.conv_low_high(low_high)
        # seg pred (fusion model)
        seg = self.seg(low_high_feature)  # 2,128,256,256
        # boudary extraction model
        boun_feat1, pred1 = self.BSM1(opt_en1)

        boun_feat2, pred2 = self.BSM2(opt_en2)
        pred2 = F.interpolate(pred2, pred1.size()[2:], mode='bilinear')
        # boun_feat2 = F.interpolate(boun_feat2, boun_feat1.size()[2:], mode='bilinear')

        boun_feat3, pred3 = self.BSM3(opt_en3)
        pred3 = F.interpolate(pred3, pred1.size()[2:], mode='bilinear')
        # boun_feat3 = F.interpolate(boun_feat3, boun_feat1.size()[2:], mode='bilinear')

        boun_feat4, pred4 = self.BSM4(opt_en4)
        pred4 = F.interpolate(pred4, pred1.size()[2:], mode='bilinear')
        # boun_feat4 = F.interpolate(boun_feat4, boun_feat1.size()[2:], mode='bilinear')

        # boun_all = torch.cat((boun_feat1, boun_feat2, boun_feat3, boun_feat4), dim=1)  # 256
        predboun = torch.cat((pred1, pred2, pred3, pred4), dim=1)
        pred_boundary = self.bounconv(predboun)
        # boun + seg
        final_seg = seg + seg * pred_boundary

        return seg, loc_seg, pred_boundary, final_seg

    def freeze_seg(self):
        # 冻结backbone
        for param in self.sar_en1.parameters():
            param.requires_grad = False
        for param in self.sar_en2.parameters():
            param.requires_grad = False
        for param in self.sar_en3.parameters():
            param.requires_grad = False
        for param in self.sar_en4.parameters():
            param.requires_grad = False
        for param in self.sar_en5.parameters():
            param.requires_grad = False
        for param in self.opt_en1.parameters():
            param.requires_grad = False
        for param in self.opt_en2.parameters():
            param.requires_grad = False
        for param in self.opt_en3.parameters():
            param.requires_grad = False
        for param in self.opt_en4.parameters():
            param.requires_grad = False
        for param in self.opt_en5.parameters():
            param.requires_grad = False
        # 冻结其他参数
        for param in self.aspp.parameters():
            param.requires_grad = False
        for param in self.low_level_down.parameters():
            param.requires_grad = False
        for param in self.sar_high_level_down.parameters():
            param.requires_grad = False
        for param in self.opt_high_level_down.parameters():
            param.requires_grad = False
        for param in self.conv_low_high.parameters():
            param.requires_grad = False
        for param in self.seg.parameters():
            param.requires_grad = False
        for param in self.locseghead.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    model = ZZNet().to(device='cuda:0')
    model.train()
    sar = torch.randn((2, 1, 512, 512), device='cuda:0')
    opt = torch.randn((2, 3, 512, 512), device='cuda:0')
    # print(model)
    output = model(sar, opt)
    print("input:", sar.shape, opt.shape)
    print("output0:", output[0].shape)
    print("output1:", output[1].shape)
    print("output2:", output[2].shape)
    print("output2:", output[3].shape)
