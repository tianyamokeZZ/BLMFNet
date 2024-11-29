import torch
import torch.nn as nn
import torch.nn.functional as F
from models.HRNet.hrnetbackbone import HRNetBackbone


class HRNet_W48(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """
    def __init__(self, num_classes=1):
        super(HRNet_W48, self).__init__()
        model = HRNetBackbone()  # 实例化 HRNetBackbone 类
        arch_net = model()  # 调用 HRNetBackbone 实例
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.backbone = arch_net
        self.num_classes = num_classes
        # extra added layers
        in_channels = 720 # 48 + 96 + 192 + 384
        self.cls_head = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x_):
        x_ = self.conv0(x_)
        x = self.backbone(x_)
        _, _, h, w = x[0].size()
        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out = self.cls_head(feats)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return out


if __name__ == "__main__":
    model = HRNet_W48(num_classes=1)
    opt = torch.randn(2, 1, 512, 512)
    print(model)
    output = model(opt)
    print("output:", output.shape)