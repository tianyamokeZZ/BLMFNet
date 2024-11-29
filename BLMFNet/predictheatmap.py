import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import os
# import pandas as pd
from PIL import Image
# import cv2 as cv
from collections import OrderedDict
import torch.nn as nn
# from datasets.DFC_track2_dataset import DFCtrack2Dataset as DFCtrack2Datasettest
from datasets.DFCtrack2_dataset_test import DFCtrack2Datasettest
from torchvision import transforms
from libs import average_meter, metric
from models.MCANet.mymodel_baseline_addattention import MCANet
from models.SOLCV7.solcv7 import SOLCV7
import matplotlib.pyplot as plt
from matplotlib import cm

img_transform = transforms.Compose([
    transforms.ToTensor()])

resore_transform = transforms.Compose([
    transforms.ToPILImage()
])


def snapshot_forward(model, dataloader, save_path):
    model.eval()
    for index, data in enumerate(dataloader):
        imgs_sar = Variable(data[0])
        imgs_opt = Variable(data[1])
        edge = Variable(data[2])
        masks = Variable(data[3])
        boundary = Variable(data[4])
        imgs_sar = imgs_sar.cuda()
        imgs_opt = imgs_opt.cuda()
        masks = masks.cuda()
        pred = model(imgs_sar, imgs_opt)
        # target_layers = [model.seg]
        # targets = [1]
        # with GradCAM(model=model, target_layers=target_layers,
        #              use_cuda=torch.cuda.is_available()) as cam:
        #     grayscale_cam = cam(input_tensor=, targets=targets)[0, :]
        heatmap_img = pred.squeeze(0)
        heatmap_img = heatmap_img.permute(1, 2, 0)
        heatmap_img = heatmap_img.cpu().data.numpy()
        plt.matshow(heatmap_img, cmap='jet', vmin=-50, vmax=100)
        # 保存图像
        image_path = os.path.join(save_path, 'mask_%d.png' % index)
        plt.savefig(image_path)


def parse_args():
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument('--test-data-root', type=str, default="/home/omnisky/ZZ/DFCtrack2/DFCtrack2_for_experiment")
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='batch size for testing (default:16)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--model-path", type=str,
                        default="/home/omnisky/ZZ/DFCtrack2/DFCtrack2_for_experiment/my_model_code_1/zznet.pth")
    parser.add_argument("--pred-path", type=str, default='/DFCtrack2_result/heat_test')
    args = parser.parse_args()
    return args


def reference():
    args = parse_args()
    dataset = DFCtrack2Datasettest(root=args.test_data_root, sync_transforms=None)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    """
    model = SOSeg(num_classes=len(class_name),
                          n_blocks=n_blocks,
                          atrous_rates=atrous_rates,
                          multi_grids=multi_grids,
                          output_stride=args.output_stride)
    """
    # model = DeepLabV3(num_classes=1)
    # model = SOLCV7(num_classes=1)
    # model = model.cuda(device='cuda:0')
    # model = RedNet(num_classes=8, one_modal_in_channel=1, two_modal_in_channel=4)
    from models.mypapermodel.mymodelv3 import ZZNet
    model = ZZNet()
    # model = MCANet(num_classes=1)
    # model = MCANet(num_classes=1)
    # model = RedNet(num_classes=1)
    state_dict = torch.load(args.model_path)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    print(state_dict.keys())
    # print(model)
    # print(state_dict)
    model.load_state_dict(state_dict)
    print('=========> load model success', args.model_path)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
    snapshot_forward(model, dataloader, args.pred_path)
    print('test done........')


if __name__ == '__main__':
    reference()
