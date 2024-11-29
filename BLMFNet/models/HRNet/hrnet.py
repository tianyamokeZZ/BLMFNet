import torch
import torch.nn as nn
import torch.nn.functional as F
from hrnetbackbone import HRNetBackbone


class HRNet_W48(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """
    def __init__(self, num_classes=1):
        super(HRNet_W48, self).__init__()
        model = HRNetBackbone()  # 实例化 HRNetBackbone 类
        arch_net = model()  # 调用 HRNetBackbone 实例
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


# if __name__ == "__main__":
#     model = HRNet_W48(num_classes=1)
#     opt = torch.randn(2, 3, 512, 512)
#     print(model)
#     output = model(opt)
#     print("output:", output.shape)

from thop import profile
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    model = HRNet_W48(num_classes=1).to("cuda")
    # model.train()
    # opt = torch.randn(1, 3, 512, 512)
    # output = model(opt)
    # print("output", output.shape)
    # print(stat(model, (3, 512, 512)))
    # input = torch.randn(1, 3, 512, 512).to("cuda")
    # flops, params = profile(model, (input, ))
    # print('FLOPs = ' + str(flops /1000**3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    # # stat(model, input_size=(3, 512, 512))
    # Params = count_parameters(model)
    # print("模型总参数量", Params)
    import time

    device = torch.device('cuda')
    model.eval()
    model.to(device)
    iterations = None
    input = torch.randn(1, 3, 512, 512).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
