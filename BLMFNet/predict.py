import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import os
import pandas as pd
from PIL import Image
# import cv2 as cv
from collections import OrderedDict
import torch.nn as nn
# from datasets.DFC_track2_dataset import DFCtrack2Dataset as DFCtrack2Datasettest
from datasets.DFCtrack2_dataset_test import DFCtrack2Datasettest
from torchvision import transforms
from libs import average_meter, metric


img_transform = transforms.Compose([
    transforms.ToTensor()])

resore_transform = transforms.Compose([
    transforms.ToPILImage()
])


def snapshot_forward(model, dataloader, save_path):
    model.eval()
    conf_mat = np.zeros((2, 2)).astype(np.int64)
    for index, data in enumerate(dataloader):
        imgs_sar = Variable(data[0])
        imgs_opt = Variable(data[1])
        masks = Variable(data[3])
        # print(imgs_sar.shape, imgs_opt.shape, masks.shape)
        imgs_sar = imgs_sar.cuda()
        imgs_opt = imgs_opt.cuda()
        masks = masks.cuda()
        outputs = model(imgs_sar, imgs_opt)
        # pred = model(imgs_sar, imgs_opt)
        # outputs = model(imgs_sar)
        preds = torch.sigmoid(outputs)
        threshold = 0.5
        preds = (preds >= threshold).int()
        preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
        masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
        conf_mat += metric.confusion_matrix(pred=preds.flatten(), label=masks.flatten(), num_classes=2)
        if masks.shape[0] == 2:
            for i in range(masks.shape[0]):
                out_preds = preds * 255

                out_masks = masks * 255
                pred_img = Image.fromarray(out_preds[i].astype(np.uint8)).convert('L')
                mask_img = Image.fromarray(out_masks[i].astype(np.uint8)).convert('L')
                pred_save_path = os.path.join(save_path, 'predict')
                mask_save_path = os.path.join(save_path, 'mask')
                path_list = [pred_save_path, mask_save_path]
                for path in range(2):
                    if not os.path.exists(path_list[path]):
                        os.makedirs(path_list[path])
                pred_img.save(os.path.join(path_list[0], 'pre_%d_%d.png' % (index, i)))
                mask_img.save(os.path.join(path_list[1], 'mask_%d_%d.png' % (index, i)))
        else:
            out_preds = preds * 255
            out_masks = masks * 255
            pred_img = Image.fromarray(out_preds.astype(np.uint8)).convert('L')
            mask_img = Image.fromarray(out_masks.astype(np.uint8)).convert('L')
            pred_save_path = os.path.join(save_path, 'predict')
            mask_save_path = os.path.join(save_path, 'mask')
            path_list = [pred_save_path, mask_save_path]
            for path in range(2):
                if not os.path.exists(path_list[path]):
                    os.makedirs(path_list[path])
            pred_img.save(os.path.join(path_list[0], 'pre_%d.png' % index))
            mask_img.save(os.path.join(path_list[1], 'mask_%d.png' % index))
    precision, recall, f1_score, iou = metric.evaluate(conf_mat)
    print("test_pre:", precision)
    print("test_rec:", recall)
    print("test f1_socre:", f1_score)
    print("test IOU:", iou)


def parse_args():
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument('--test-data-root', type=str, default="/home/omnisky/ZZ/DFCtrack2/DFCtrack2_for_experiment")
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='batch size for testing (default:16)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--model-path", type=str,
                        default="/home/omnisky/ZZ/DFCtrack2/DFCtrack2_for_experiment/my_model_code_1/zz1net.pth")
    parser.add_argument("--pred-path", type=str, default="/home/omnisky/ZZ/DFCtrack2/DFCtrack2_for_experiment/zz1netresult")
    args = parser.parse_args()
    return args


def reference():
    args = parse_args()
    dataset = DFCtrack2Datasettest(root=args.test_data_root, sync_transforms=None)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8)
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
    from models.mypapermodel.mymodelv4 import ZZ1Net
    model = ZZ1Net()
    # model = MCANet(num_classes=1)
    # model = MCANet(num_classes=1)
    # model = RedNet(num_classes=1)
    state_dict = torch.load(args.model_path, map_location='cuda:0')
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