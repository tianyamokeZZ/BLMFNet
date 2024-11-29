import torch
import torch.nn.functional as F
from torch import nn
from models.mypapermodel.aspp import _ASPP
from models.mypapermodel.mymodule import AFF, BSM
from models.mypapermodel.myAFD import AFD


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


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2, padding=3),
        )

    def forward(self, x):
        return self.decode(x)


class ZZ0Net(nn.Module):
    def __init__(self, num_classes, atrous_rates=[6,12,18], channel2=256, channel1=128 ,channel0=64, num_class=1):
        super(ZZ0Net, self).__init__()
        self.sar_en1 = _EncoderBlock(1, 64) # 256->128, 1->64
        self.sar_en2 = _EncoderBlock(64, 256)  # 128->64, 64->256
        self.sar_en3 = _EncoderBlock(256, 512)  # 64->32, 256->512
        self.sar_en4 = _EncoderBlock(512, 1024, downsample=False)  # 32->32 *** , 512->1024
        self.sar_en5 = _EncoderBlock(1024, 2048, downsample=False)  # 32->32 *** , 1024->2048

        self.opt_en1 = _EncoderBlock(3, 64) # 256->128, 4->64
        self.opt_en2 = _EncoderBlock(64, 256)  # 128->64, 64->256
        self.opt_en3 = _EncoderBlock(256, 512)  # 64->32, 256->512
        self.opt_en4 = _EncoderBlock(512, 1024, downsample=False)  # 32->32 *** , 512->1024
        self.opt_en5 = _EncoderBlock(1024, 2048, downsample=False)  # 32->32 *** , 1024->2048
        self.aspp = nn.Sequential(
            _ASPP(256 * 2, 256, atrous_rates),
            nn.Conv2d(256 * 5, 256, kernel_size=1, stride=1, padding=0))
        self.low_level_down = nn.Conv2d(256 * 2, 48, kernel_size=1, stride=1, padding=0)
        self.sar_high_level_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.opt_high_level_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.conv_low_high = nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1)
        self.seg = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
        self.locseghead = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )
        self.AFF = AFF(channels=256)
        self.seg_feature = None
        initialize_weights(self)

    def forward(self, sar, opt):
        sar_en1 = self.sar_en1(sar)
        sar_en2 = self.sar_en2(sar_en1)
        sar_en3 = self.sar_en3(sar_en2)
        sar_en4 = self.sar_en4(sar_en3)
        sar_en5 = self.sar_en5(sar_en4)
        opt_en1 = self.opt_en1(opt)
        opt_en2 = self.opt_en2(opt_en1)
        opt_en3 = self.opt_en3(opt_en2)
        opt_en4 = self.opt_en4(opt_en3)
        opt_en5 = self.opt_en5(opt_en4)
        low_level_features = self.low_level_down(torch.cat([sar_en2, opt_en2], 1))  # 2,512,128,128 ->2,48,128,128
        x0, x1 = self.AFF(self.sar_high_level_down(sar_en5), self.opt_high_level_down(opt_en5))
        high_level_features = torch.cat([x0, x1], 1)
        loc_seg = self.locseghead(high_level_features)  # 输入 2,512,64,64
        high_level_features = self.aspp(high_level_features)  # 2,512,64,64 多尺度特征
        #  upsample被替换为 interpolate
        high_level_features = F.interpolate(high_level_features, sar_en2.size()[2:], mode='bilinear')  # 2,256,128,128
        low_high = torch.cat([low_level_features, high_level_features], 1)  # 2,256+48,128,128
        self.seg_feature = self.conv_low_high(low_high)
        seg = self.seg(self.seg_feature)  # 2,128,256,256
        return seg, loc_seg

    def get_seg_feature(self):
        return self.seg_feature


class ZZNet(nn.Module):
    def __init__(self):
        super(ZZNet, self).__init__()
        ori_model = ZZ0Net(num_classes=1)
        state_dict = {k.replace('module.', ''): v for k, v in torch.load("/home/omnisky/ZZ/DFCtrack2/DFCtrack2_for_experiment/my_model_code_1/zz0net.pth").items()}
        ori_model.load_state_dict(state_dict)
        # self.stage1 = ori_model
        self.opt_en1 = ori_model.opt_en1
        self.opt_en2 = ori_model.opt_en2
        self.opt_en3 = ori_model.opt_en3
        self.opt_en4 = ori_model.opt_en4
        for pram in self.opt_en1.parameters():
            pram.requires_grad = False
        for pram in self.opt_en2.parameters():
            pram.requires_grad = False
        for pram in self.opt_en3.parameters():
            pram.requires_grad = False
        for pram in self.opt_en4.parameters():
            pram.requires_grad = False
        self.BSM1 = BSM(all_channel=64)
        self.BSM2 = BSM(all_channel=256)
        self.BSM3 = BSM(all_channel=512)
        self.BSM4 = BSM(all_channel=1024)
        self.bounconv = nn.Conv2d(256, 1, kernel_size=1, padding=0, dilation=1, bias=True)
        self.boun_feature = None

    def forward(self, sar, opt):
        opten1 = self.opt_en1(opt)
        opten2 = self.opt_en2(opten1)
        opten3 = self.opt_en3(opten2)
        opten4 = self.opt_en4(opten3)
        feat1, pre_boundary1 = self.BSM1(opten1)
        feat2, pre_boundary2 = self.BSM2(opten2)
        feat3, pre_boundary3 = self.BSM3(opten3)
        feat4, pre_boundary4 = self.BSM4(opten4)

        feat2 = F.interpolate(feat2, feat1.size()[2:], mode='bilinear')
        feat3 = F.interpolate(feat3, feat1.size()[2:], mode='bilinear')
        feat4 = F.interpolate(feat4, feat1.size()[2:], mode='bilinear')

        self.boun_feature = torch.cat((feat1, feat2, feat3, feat4), dim=1)
        pred_boudary = self.bounconv(self.boun_feature)
        pred_boudary = F.interpolate(pred_boudary, opt.size()[2:], mode='bilinear')
        return pred_boudary

    def get_boun_feature(self):
        return self.boun_feature


class ZZ1Net(nn.Module):
    def __init__(self):
        super(ZZ1Net, self).__init__()
        ori_model = ZZ0Net(num_classes=1)
        state_dict = {k.replace('module.', ''): v for k, v in torch.load("/home/omnisky/ZZ/DFCtrack2/DFCtrack2_for_experiment/my_model_code_1/zz0net.pth").items()}
        ori_model.load_state_dict(state_dict)
        self.fusion = ori_model
        for pram in self.fusion.parameters():
            pram.requires_grad = False
        ori1_model = ZZNet()
        state_dict = {k.replace('module.', ''): v for k, v in torch.load(
            "/home/omnisky/ZZ/DFCtrack2/DFCtrack2_for_experiment/my_model_code_1/zznet.pth").items()}
        ori1_model.load_state_dict(state_dict)
        self.boundaryextraction = ori1_model
        self.simpledecoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.AFD = AFD(s_channels=256, c_channels=256)
        self.seg = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, sar, opt):
        self.fusion(sar, opt)
        seg_feature = self.fusion.get_seg_feature()  # 2, 256, 128, 128
        self.boundaryextraction(sar, opt)
        boun_feature = self.boundaryextraction.get_boun_feature()  # 2 256 256 256
        seg_feature = self.simpledecoder(seg_feature)
        final_feature = self.AFD(seg_feature, boun_feature)
        seg_final = self.seg(final_feature)
        return seg_final


if __name__ == "__main__":
    model = ZZ1Net().cuda()
    model.train()
    sar = torch.randn(2, 1, 512, 512).cuda()
    opt = torch.randn(2, 3, 512, 512).cuda()
    # print(model)
    output = model(sar, opt)
    print("input:", sar.shape, opt.shape)
    print("output0:", output.shape)
    # print("output0:", output[0].shape)
    # print("output1:", output[1].shape)
    # print("output2:", output[2].shape)

