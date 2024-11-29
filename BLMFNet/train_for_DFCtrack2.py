import argparse
# from DFC_track2_dataset import DFCtrack2Dataset
from datasets.DFC_track2_dataset import DFCtrack2Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from libs import average_meter, metric
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


def parse_args():
    parser = argparse.ArgumentParser(description="Remote Sensing Segmentation by PyTorch")
    # dataset 相关的输入
    parser.add_argument('--train-data-root', type=str, default="/home/omnisky/ZZ/DFCtrack2/DFCtrack2_for_experiment")
    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--train-batch-size', type=int, default=4, metavar='N', help='batch size for training (default:16)')
    # learning_rate学习率相关的设置
    parser.add_argument('--total-epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 120)')
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    # 加载的model
    parser.add_argument('--model', type=str, default='mypapermodel', help='model name')
    # -===================！！！！！！！
    parser.add_argument('--save-pseudo-data-path', type=str, default='pseudo-data')
    # criterion
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch (default:0)')
    # loss
    parser.add_argument('--classes-weight', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default:0.9)')
    # optimizer
    parser.add_argument('--optimizer-name', type=str, default='Adam')
    # environment
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=2, help='numbers of GPUs')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    # 创建一个目录保存结果
    if args.use_cuda:
        print('Numbers of GPUs:', len(args.gpu_ids))
    else:
        print("Using CPU")
    return args


def dice_loss(predict, mask, ep=1e-8):
    predict = predict.sigmoid()
    predict = predict.flatten(1)
    mask = mask.flatten(1).float()
    intersection = 2*torch.sum(predict*mask)+ep
    union = torch.sum(predict)+torch.sum(mask)+ep
    loss = 1 - intersection/union
    return loss


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.train_dataset = DFCtrack2Dataset(root=args.train_data_root) # random flip 不设置sync为None就是进行增强
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
        print('Number samples {}.'.format(len(self.train_dataset)))
        self.criterion_semantic = nn.BCEWithLogitsLoss(reduction='mean').to(device=args.device)
        self.criterion_loc_semantic = nn.BCEWithLogitsLoss(reduction='mean').to(device=args.device)
        self.criterion_boundary = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([10])).to(device=args.device)
        self.criterion_final_seg = nn.BCEWithLogitsLoss(reduction='mean').to(device=args.device)
        device = args.device
        # 加载模型
        if args.model == 'mypapermodel':
            from models.mypapermodel.mymodelv2 import ZZ1Net
            model = ZZ1Net()
            print('======> model mypapermodel =============== ')
        # 训练用什么
        if args.use_cuda:
            model = model.to(device=device)
            self.model = nn.DataParallel(model, device_ids=args.gpu_ids)
        # SGD不work，Adadelta出奇的好？
        if args.optimizer_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
        if args.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=1e-4)
        if args.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params=model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
        self.max_iter = args.total_epochs * len(self.train_loader)

    def training_stage1(self, epoch):
        self.model.train()  # 把module设成训练模式，对Dropout和BatchNorm有影响
        train_loss = average_meter.AverageMeter()
        # 放置混淆矩阵，就2分类
        conf_mat = np.zeros((2, 2)).astype(np.int64)
        tbar = tqdm(self.train_loader)
        for index, data in enumerate(tbar):
            imgs_sar = Variable(data[0]).to(device=args.device)
            imgs_opt = Variable(data[1]).to(device=args.device)
            masks = Variable(data[3]).to(device=args.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs_sar, imgs_opt)
            semantic = outputs[0]
            loc_semantic = outputs[1]
            # torch.max(tensor, dim)：指定维度上最大的数，返回tensor和下标
            bce_loss = self.criterion_semantic(semantic, masks)
            dic_loss = dice_loss(semantic, masks)
            loc_loss = self.criterion_loc_semantic(loc_semantic, masks)
            loss = bce_loss + dic_loss + loc_loss
            train_loss.update(loss, self.args.train_batch_size)
            loss.backward()
            self.optimizer.step()
            tbar.set_description('epoch {}/{}, loss {}, BCELoss {} , with learning rate {}.'
                                 .format(epoch, args.total_epochs, train_loss.avg, bce_loss,
                                         self.optimizer.state_dict()['param_groups'][0]['lr']))
            # predict and mask convert to uint8 to get conf-mat
            preds = torch.sigmoid(outputs[0])
            threshold = 0.5
            preds = (preds >= threshold).int()
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            conf_mat += metric.confusion_matrix(pred=preds.flatten(), label=masks.flatten(), num_classes=2)
        build_precision, build_recall, build_f1_score, build_iou = metric.evaluate(conf_mat)
        print("build precision:", build_precision)
        print("build recall:", build_recall)
        print("build_f1_score:", build_f1_score)
        print("build_iou:", build_iou)
        torch.save(self.model.state_dict(), "mypapermodel.pth")

    def training_stage2(self, epoch):
        # 冻结相关参数
        self.model.module.freeze_seg()
        self.model.train()  # 把module设成训练模式，对Dropout和BatchNorm有影响
        train_loss = average_meter.AverageMeter()
        # 放置混淆矩阵，就2分类
        conf_mat = np.zeros((2, 2)).astype(np.int64)
        tbar = tqdm(self.train_loader)
        for index, data in enumerate(tbar):
            imgs_sar = Variable(data[0]).to(device=args.device)
            imgs_opt = Variable(data[1]).to(device=args.device)
            boundary = Variable(data[4]).to(device=args.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs_sar, imgs_opt)
            pred_boudary = outputs[2]
            # torch.max(tensor, dim)：指定维度上最大的数，返回tensor和下标
            bce_loss = self.criterion_boundary(boundary, pred_boudary)
            dic_loss = dice_loss(boundary, pred_boudary)
            loss = bce_loss + dic_loss
            train_loss.update(loss, self.args.train_batch_size)
            loss.backward()
            self.optimizer.step()
            tbar.set_description('epoch {}/{}, loss {}, BCELoss {} , with learning rate {}.'
                                 .format(epoch, args.total_epochs, train_loss.avg, bce_loss,
                                         self.optimizer.state_dict()['param_groups'][0]['lr']))
            # predict and mask convert to uint8 to get conf-mat
            preds = torch.sigmoid(outputs[2])
            threshold = 0.5
            preds = (preds >= threshold).int()
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            masks = boundary.data.cpu().numpy().squeeze().astype(np.uint8)
            conf_mat += metric.confusion_matrix(pred=preds.flatten(), label=masks.flatten(), num_classes=2)
        build_precision, build_recall, build_f1_score, build_iou = metric.evaluate(conf_mat)
        print("boundary precision:", build_precision)
        print("boundary recall:", build_recall)
        print("boundary_f1_score:", build_f1_score)
        print("boundary_iou:", build_iou)
        torch.save(self.model.state_dict(), "mypapermodel.pth")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    print(args)
    # writer = SummaryWriter()
    trainer = Trainer(args)
    scheduler1 = StepLR(trainer.optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(args.start_epoch, args.total_epochs):
        trainer.training_stage1(epoch)
        scheduler1.step()
    scheduler2 = StepLR(trainer.optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(args.start_epoch, args.total_epochs):
        trainer.training_stage2(epoch)
        scheduler2.step()
