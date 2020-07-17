# encoding: utf-8
"""
 @project:DogCatProject
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/10/19 10:13 AM
 @desc:
        在模型定义、数据处理和训练等过程都有很多变量，这些变量应提供默认值，并统一放置在配置文件中.
        这样在后期调试、修改代码或迁移程序时会比较方便，在这里我们将所有可配置项放在此文件中
"""

import warnings


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'ResNet34'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = '/home/jh/DogCat/train'  # 训练集存放路径
    test_data_root = '/home/jh/DogCat/test'  # 测试集存放路径
    load_model_path = None  # 加载预训练模型的路径，如checkpoints/model.pth，为None代表不加载模型
    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'  # 存储测试集的结果

    batch_size = 64  # 显存不够，就把batch_size调低
    use_gpu = True
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # 每20个batch打印一次训练结果

    max_epoch = 10
    lr = 0.001
    lr_decay = 0.95  # 学习率的衰减率, lr *= lr_decay
    weight_decay = 1e-4  # 损失函数

    def parse(self, kwargs):
        """
        根据字典kwargs更新config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        # 输出配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

# DefaultConfig.parse = parse
# opt = DefaultConfig()
