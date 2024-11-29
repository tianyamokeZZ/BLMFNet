from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn

# backbone = IntermediateLayerGetter(
#     resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True]),
#     return_layers={'layer4': 'stage4'}
# )
#
# # test
# x = torch.randn(3, 3, 224, 224).cpu()
# result = backbone(x)
# for k, v in result.items():
#     print(k, v.shape)


class PositionAttention(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttention, self).__init__()
        self.convB = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convC = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        # 创建一个可学习参数a作为权重,并初始化为0.
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b ,c ,h ,w = x.size()
        B = self.convB(x)
        C = self.convB(x)
        D = self.convB(x)
        S = self.softmax(torch.matmul(B.view(b, c, h* w).transpose(1, 2), C.view(b, c, h * w)))
        E = torch.matmul(D.view(b, c, h * w), S.transpose(1, 2)).view(b, c, h, w)
        # gamma is a parameter which can be training and iter
        E = self.gamma * E + x

        return E


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        X = self.softmax(torch.matmul(x.view(b, c, h * w), x.view(b, c, h * w).transpose(1, 2)))
        X = torch.matmul(X.transpose(1, 2), x.view(b, c, h * w)).view(b, c, h, w)
        X = self.beta * X + x
        return X


class DAHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DAHead, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, num_classes, kernel_size=3, padding=1, bias=False),
        )

        self.PositionAttention = PositionAttention(in_channels // 4)
        self.ChannelAttention = ChannelAttention()

    def forward(self, x):
        x_PA = self.conv1(x)
        x_CA = self.conv2(x)
        PosionAttentionMap = self.PositionAttention(x_PA)
        ChannelAttentionMap = self.ChannelAttention(x_CA)
        # 这里可以额外分别做PAM和CAM的卷积输出,分别对两个分支做一个上采样和预测,
        # 可以生成一个cam loss和pam loss以及最终融合后的结果的loss.以及做一些可视化工作
        # 这里只输出了最终的融合结果.与原文有一些出入.
        output = self.conv3(PosionAttentionMap + ChannelAttentionMap)
        output = nn.functional.interpolate(output, 512, mode="bilinear", align_corners=True)
        output = self.conv4(output)
        return output


class DAnet(nn.Module):
    def __init__(self, num_classes):
        super(DAnet, self).__init__()
        self.ResNet50 = IntermediateLayerGetter(
            resnet50(weights=None, replace_stride_with_dilation=[False, True, True]),
            return_layers={'layer4': 'stage4'}
        )
        self.decoder = DAHead(in_channels=2048, num_classes=num_classes)

    def forward(self, x):
        feats = self.ResNet50(x)
        # self.ResNet50返回的是一个字典类型的数据.
        x = self.decoder(feats["stage4"])
        return x


# if __name__ == "__main__":
#     x = torch.randn(1, 3, 512, 512)
#     model = DAnet(num_classes=1)
#     result = model(x)
#     print(result.shape)

from thop import profile
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    model = DAnet(num_classes=1).to("cuda")
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
