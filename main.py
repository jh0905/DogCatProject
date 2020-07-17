# encoding: utf-8
"""
 @project:DogCatProject
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/10/19 10:13 AM
 @desc:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
import models
from config import DefaultConfig
from data.dataset import DogCat
from utils.visualize import Visualizer

opt = DefaultConfig()


def train(**kwargs):
    opt.parse(kwargs)  # 根据命令行输入更新配置，如 python main.py train --env='env1219' --
    vis = Visualizer(opt.env)

    # 第一步：加载模型（模型，预训练参数，GPU）
    model = getattr(models, opt.model)()  # 等价于 models.AlexNet()
    # model = torchvision.models.resnet34(pretrained=True, num_classes=1000)
    # model.fc = nn.Linear(512, 2)  # 修改最后一层为我们的二分类问题
    if opt.load_model_path:
        model.load(opt.load_model_path)  # 加载模型参数
    if opt.use_gpu:
        model.cuda()
    # 第二步：加载数据（训练集、验证集，用DataLoader来装载）
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_data_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # 第三步：定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # 第四步：统计指标：平滑处理之后的损失，混淆矩阵
    # meter是PyTorchNet里面的一个重要工具，可以帮助用户快速统计训练过程中的一些指标。
    loss_meter = meter.AverageValueMeter()  # loss_meter.value()，返回一个二元组，第一个元素是均值，第二个元素是标准差
    confusion_matrix = meter.ConfusionMeter(2)  # confusion_matrix是一个2*2的混淆矩阵
    previous_loss = 1e100  # 初始置为10^100

    # 第五步：开始训练
    for epoch in range(opt.max_epoch):
        loss_meter.reset()  # 置为nan
        confusion_matrix.reset()  # 清零
        # tqdm 是一个快速，可扩展的Python进度条，可以在长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
        import math
        for ii, (data, label) in tqdm(enumerate(train_data_loader), total=math.ceil(len(train_data) / opt.batch_size)):
            # 模型参数训练
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()  # 每轮都要清空一轮梯度信息
            pred = model(input)
            # 计算损失，输入为(pred,target)，其中pred的shape为(batch_size,n_classes)，target为(batch_size)，target中的值为0或1
            # 示例
            # pred = t.randn(128,2)
            # target = t.empty(128,dtype=t.long).random_(2)
            # criterion(pred,target)
            # 某个样本的预测值x为[0.11,0.39]，对应标签y为0，那么计算损失的方式为： -x[y] + log(exp(x[0])+exp(x[1]))
            # 具体计算方式见: https://blog.csdn.net/geter_CS/article/details/84857220

            loss = criterion(pred, target)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新网络权重参数
            # 更新统计指标以及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(pred.data, target.data)

            if ii % opt.print_freq == opt.print_freq - 1:  # 每20个batch输出一次损失
                vis.plot('loss', loss_meter.value()[0])
                # 如果需要的话，进入debug模式
                # if os.path.exists(opt.debug_file):
                #     ipdb.set_trace()

        # 保存训练好的模型
        model.save()  # 存在checkpoints目录下

        # 在验证集上测试结果并可视化
        val_cm, val_accuracy = val(model, val_data_loader)
        vis.plot('val_accuracy', val_accuracy)  # 绘制的是精确度

        vis.log('epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}'.format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr
        ))

        # 如果损失不再下降，则衰减学习率
        if loss_meter.value()[0] > previous_loss:
            lr = opt.lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]  # 更新previous_loss的值为当前损失的平均值


def val(model, data_loader):
    """
    计算模型在验证集上的准确率等信息，用以辅助训练
    """
    model.eval()  # 把模型设为推理模式，不会计算梯度信息
    confusion_matrix = meter.ConfusionMeter(2)  # 2 × 2的混淆矩阵
    for ii, data in enumerate(data_loader):
        input, label = data
        val_input = Variable(input)
        val_label = Variable(label)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        pred = model(val_input)
        confusion_matrix.add(pred.data, val_label.data)

    model.train()  # 把模型恢复为训练模式

    cm_value = confusion_matrix.value()  # 返回一个2*2的ndarray数组
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())  # 对角线是预测正确的元素个数
    return confusion_matrix, accuracy


def test(**kwargs):
    """
    测试
    """
    opt.parse(kwargs)
    # ipdb.set_trace() # 设置断点，进行调试
    # 配置模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # 准备测试集
    test_data = DogCat(opt.test_data_root, test=True)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, test_id) in enumerate(test_data_loader):
        input = Variable(data)
        if opt.use_gpu:
            input = input.cuda()
        pred = model(input)
        probability = F.softmax(pred)[:, 0].data.tolist()
        batch_results = [(test_id_.item(), probability_) for test_id_, probability_ in zip(test_id, probability)]
        results += batch_results
    write_csv(results, opt.result_file)
    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def help():
    """
    打印帮助的信息:python file.py help
    """
    print('''
        usage : python file.py <function> [--args=value]
        <function> := train | test | help
        example: 
                python {0} train --env='env0701' --lr=0.01
                python {0} test --dataset='path/to/dataset/root/'
                python {0} help
        avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()
