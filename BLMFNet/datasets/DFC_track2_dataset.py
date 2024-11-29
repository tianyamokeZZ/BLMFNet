from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms.functional as F
from torchvision import transforms
import tifffile
import numpy as np
import torch
import cv2


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

# mask的变换
mask_transform = MaskToTensor()
# 光学图像的均值和标准差
mean_optical = [81.37792521, 88.03442615, 72.09852648]
std_optical = [45.62213273, 41.71990951, 42.88563375]
# 定义光学图像的变换
img_opt_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_optical, std=std_optical)
])
# SAR图像的均值和标准差
mean_sar = [0.26343689]
std_sar = [0.59567231]
# 定义SAR图像的变换
img_sar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_sar, std=std_sar)
])
# edge的变换
edge_transform = transforms.ToTensor()


class SyncTransform:
    def __init__(self, rand=0.5):
        self.rand = rand

    def random_flip(self, img_sar, img_opt, edge, mask, boundary_ture):
        # 随机水平翻转
        if torch.rand(1).item() > 0.5:
            img_sar = F.hflip(img_sar)
            img_opt = F.hflip(img_opt)
            edge = F.hflip(edge)
            mask = F.hflip(mask)
            boundary_ture = F.hflip(boundary_ture)
        # 随机垂直翻转
        if torch.rand(1).item() > 0.5:
            img_sar = F.vflip(img_sar)
            img_opt = F.vflip(img_opt)
            edge = F.vflip(edge)
            mask = F.vflip(mask)
            boundary_ture = F.vflip(boundary_ture)
        return img_sar, img_opt, edge, mask, boundary_ture

    def __call__(self, img_sar, img_opt, edge, mask, boundary_ture):
        # 随机翻转
        img_sar, img_opt, edge, mask, boundary_ture = self.random_flip(img_sar, img_opt, edge, mask, boundary_ture)
        return img_sar, img_opt, edge, mask, boundary_ture
transform = SyncTransform()


class DFCtrack2Dataset(Dataset):
    def __init__(self, root, sync_transforms=transform):
        # 数据相关
        self.img_sar_transform = img_sar_transform
        self.img_opt_transform = img_opt_transform
        self.edge_transform = edge_transform
        self.mask_transform = mask_transform
        self.boundary = mask_transform
        self.sync_transform = sync_transforms
        self.sync_img_mask = []
        img_sar_dir = os.path.join(root, 'sar')
        img_opt_dir = os.path.join(root, 'rgb')
        img_edge_dir = os.path.join(root, 'sober')
        # mask_dir = os.path.join(root, 'train_dataset')
        mask_dir = os.path.join(root, 'train_dataset')
        img_boundary_dir = os.path.join(root, 'boundary_chu')
        for img_filename in os.listdir(mask_dir):
            img_mask_pair = (os.path.join(img_sar_dir, img_filename),
                             os.path.join(img_opt_dir, img_filename),
                             os.path.join(mask_dir, img_filename),
                             os.path.join(img_edge_dir, img_filename.replace('.tif', '.png')),
                             os.path.join(img_boundary_dir, img_filename.replace('.tif', '.png'))
            )
            self.sync_img_mask.append(img_mask_pair)
        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        img_sar_path, img_opt_path, mask_path, edge_path, boundary_path = self.sync_img_mask[index]
        # tiff 文件用tiffile读取
        img_sar = tifffile.imread(img_sar_path).astype(np.float32)
        img_opt = tifffile.imread(img_opt_path).astype(np.float32)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
        boundary_ture = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
        # 将标签中的255转换为1 不知道为什么读进来是-1
        mask[mask == 255] = 1
        boundary_ture[boundary_ture == 255] = 1
        # transform
        img_sar = self.img_sar_transform(img_sar)
        img_opt = self.img_opt_transform(img_opt)
        edge = self.edge_transform(edge)
        mask = self.mask_transform(mask)
        boundary_ture = self.boundary(boundary_ture)
        # 将 mask 转换为长整型（long）添加一个新的维度
        mask = mask.float().unsqueeze(0)
        boundary_ture = boundary_ture.float().unsqueeze(0)
        if self.sync_transform is not None:
            img_sar, img_opt, edge, mask, boundary_ture = self.sync_transform(img_sar, img_opt, edge, mask, boundary_ture)
        return img_sar, img_opt, edge, mask, boundary_ture

    def __len__(self):
        return len(self.sync_img_mask)


if __name__ ==  "__main__":
    isbi_dataset = DFCtrack2Dataset(root="/home/omnisky/ZZ/DFCtrack2/DFCtrack2_for_experiment")  # 数据
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for img_sar, img_opt, edge, mask, boundary in train_loader:
        print(img_sar.shape)
        print(img_opt.shape)
        print(edge.shape)
        print(mask.shape)
        print(boundary.shape)
