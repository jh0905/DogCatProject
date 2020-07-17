# encoding: utf-8
"""
 @project:DogCatProject
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/12/19 9:56 AM
 @desc: 对Dataset类进行封装，在具体使用时可通过DataLoader来加载数据

         Kaggle提供的猫狗数据集包括训练集和测试集，而我们在实际使用中，还需专门从训练集中取出一部分作为验证集。
        对于这三类数据集，其相应操作也不太一样，而如果专门写三个Dataset，则稍显复杂和冗余，因此这里通过加一些判断来区分。
        对于训练集，我们希望做一些数据增强处理，如随机裁剪、随机翻转、加噪声等，而验证集和测试集则不需要。
"""

import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        """
        :param root: 数据集所在目录的路径，例如： data/test 或 data/train
        :param transforms: 图像转换操作（数据增广），只针对训练集，验证集和测试集不需要增广
        :param train: 是否为训练集
        :param test: 是否为测试集
        """
        self.test = test
        images = [os.path.join(root, img) for img in os.listdir(root)]
        # 提取文件路径中的编号，对图像排序
        if self.test:  # 测试集文件路径：'/home/jh/DogCat/data/test/516.jpg'
            images = sorted(images, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:  # 训练集'/home/jh/DogCat/data/train/dog.12419.jpg'
            images = sorted(images, key=lambda x: int(x.split('.')[-2]))

        # shuffle images
        np.random.seed(2020)
        images = np.random.permutation(images)
        imgs_num = len(images)
        # 划分训练集（训练：验证=7:3）
        if self.test:
            self.images = images
        elif train:
            self.images = images[:int(0.7 * imgs_num)]
        else:
            self.images = images[int(0.7 * imgs_num):]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 三个通道，所以三组值
            if self.test or not train:  # 测试集和验证集不需要增强
                self.transforms = T.Compose([
                    T.Resize(224),  # 按照原来的宽高比,缩放图像中的短边至224
                    T.CenterCrop(224),  # 中间截取224*224
                    T.ToTensor(),  # image 转 tensor, 像素值从0~255 归一化到[0., 1.]
                    normalize])  # 标准化
            else:
                self.transforms = T.Compose(
                    [T.Resize(256),
                     T.RandomResizedCrop(224),  # 随机裁剪
                     T.RandomHorizontalFlip(),  # 随机水平翻转
                     T.ToTensor(),
                     normalize])

    def __getitem__(self, idx):
        """
        将文件读取等费时操作放在此函数中，利用多进程加速。
        :param idx:图片索引
        :return: 对于训练集，返回一张图片对象和标签；测试集返回一张图片对象和其编号（如1000.jpg返回100）
        """
        img_path = self.images[idx]
        if self.test:
            label = int(self.images[idx].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        image = self.transforms(Image.open(img_path))
        return image, label

    def __len__(self):
        """
        :return: 返回训练集或验证集或测试集中所有图片的个数
        """
        return len(self.images)
