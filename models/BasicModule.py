# encoding: utf-8
"""
 @project:DogCatProject
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/12/19 11:10 AM
 @desc: BasicModule是对nn.Module的简易封装，提供快速加载和保存模型的接口

        关于模型定义的注意事项：
        (1)尽量使用nn.Sequential（比如AlexNet）
        (2)将经常使用的结构封装成子Module（比如GoogLeNet的Inception结构，ResNet的Residual Block结构）
        (3)将重复且有规律性的结构，用函数生成（比如VGG的多种变体，ResNet多种变体都是由多个重复卷积层组成）
"""
import torch
import torch.nn as nn
import time


class BasicModule(nn.Module):
    """
    封装了nn.Module，主要提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 模型的默认名字

    def load(self, path):
        """
        :param path: 模型的路径
        :return: 返回指定路径的模型
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        如AlexNet_1210_20:20:29.pth
        """
        if not name:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime((prefix + '%m%d_%H:%M:%S.pth'))
        torch.save(self.state_dict(), name)
        return name
