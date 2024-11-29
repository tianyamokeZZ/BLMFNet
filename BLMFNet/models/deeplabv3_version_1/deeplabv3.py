import torch.nn as nn
import torch.nn.functional as F
from models.deeplabv3_version_1.resnet import ResNet50
from models.deeplabv3_version_1.aspp import ASPP_Bottleneck
import torch


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=16):
        super(DeepLabV3, self).__init__()
        self.num_classes = num_classes
        self.resnet = ResNet50()
        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        feature_map = self.resnet(x)
        output = self.aspp(feature_map)
        output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=False)
        return output


if __name__ == '__main__':
    model = DeepLabV3()
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    print(model)
    output = model(image)
    print("input:", image.shape)
    print("output:", output.shape)


# from thop import profile
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# if __name__ == '__main__':
#     model = DeepLabV3().to("cuda")
#     # model.train()
#     # opt = torch.randn(1, 3, 512, 512)
#     # output = model(opt)
#     # print("output", output.shape)
#     # print(stat(model, (3, 512, 512)))
#     input = torch.randn(1, 3, 512, 512).to("cuda")
#     flops, params = profile(model, (input, ))
#     print('FLOPs = ' + str(flops /1000**3) + 'G')
#     print('Params = ' + str(params / 1000 ** 2) + 'M')
#     # stat(model, input_size=(3, 512, 512))
#     Params = count_parameters(model)
#     print("模型总参数量", Params)
