import copy
import numpy as np
import pandas as pd
import torchvision.models
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import time

import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator
import timm

if __name__ == "__main__":

    # 这里面的变量都相当于全局变量 ！！

    # GPU计算
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #  训练总轮数
    total_epochs = 100
    # 每次取出样本数
    batch_size = 32
    # 初始学习率
    Lr = 0.1

    SAVE_PATH = './result/'
    filename = '{}best_mobilev3_model'.format(SAVE_PATH)  # 文件扩展名在保存时添加

    data_flag = 'bloodmnist'
    download = True
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    CLASSES = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    torch.backends.cudnn.benchmark = True #TODO Ture

    # 准备数据
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
             # , transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            , transforms.RandomCrop(28, padding=4)  # 先四周填充0，在吧图像随机裁剪成32*32
            , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
            , transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 均值，标准差
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
             # , transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            , transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }
    # 准备数据 这里将训练集和验证集写到了一个list里 否则后面的训练与验证阶段重复代码太多
    # load the data
    image_datasets = {
        x: DataClass(split=x,
                   transform=data_transforms[x], download=True) for x in ['train', 'test']}

    dataloaders: dict = {
        x: data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False
        ) for x in ['train', 'test']
    }

    # 定义模型
    model_ft = timm.create_model('convnext_xlarge', pretrained=False, num_classes=CLASSES)
    # model_ft = torchvision.models.(pretrained=False, num_classes=CLASSES)
    print(model_ft)
    total_params_o = sum(p.numel() for p in model_ft.parameters())
    print("模型的参数量：", total_params_o)


    print(model_ft)
    # 计算的参数量
    total_params_o = sum(p.numel() for p in model_ft.parameters())
    print("模型的参数量：", total_params_o)

    model_ft.to(device)
    # 创建损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # 训练模型
    # 显示要训练的模型
    print("==============当前模型要训练的层==============")
    from thop import profile
    input = torch.zeros((1, 3, 28, 28)).to(device)
    flops, params = profile(model_ft.to(device), inputs=(input,))

    print("FLOPS：", flops/1000000)
    from torchstat import stat
    stat(model_ft, (3, 28, 28))

    # 训练模型所需参数
    # 用于记录损失值未发生变化batch数
    counter = 0
    # 记录训练次数
    total_step = {
        'train': 0, 'test': 0
    }
    # 记录开始时间
    since = time.time()
    # 记录当前最小损失值
    test_loss_min = np.Inf
    # 保存模型文件的尾标
    save_num = 0
    # 保存最优正确率
    best_acc = 0

    accnp = []
    lossnp = []
    for epoch in range(total_epochs):
        # 动态调整学习率
        if counter / 10 == 1:
            counter = 0
            Lr = Lr * 0.7

        # 在每个epoch里重新创建优化器？？？
        optimizer = optim.SGD(model_ft.parameters(), lr=Lr, momentum=0.9, weight_decay=5e-4)

        print('Epoch {}/{}'.format(epoch + 1, total_epochs))
        print('-' * 10)
        print()
        # 训练和验证 每一轮都是先训练train 再验证test
        for phase in ['train', 'test']:
            # 调整模型状态
            if phase == 'train':
                model_ft.train()  # 训练
            else:
                model_ft.eval()  # 验证

            # 记录损失值
            running_loss = 0.0
            # 记录正确个数
            running_corrects = 0

            # 一次读取一个batch里面的全部数据
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)

                labels = labels.squeeze(dim=1).long().to(device)

                # 梯度清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    loss = loss_fn(outputs, labels)

                    # torch.max() 返回的是一个元组 第一个参数是返回的最大值的数值 第二个参数是最大值的序号
                    _, preds = torch.max(outputs, 1)  # 前向传播 这里可以测试 在test时梯度是否变化

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()  # 反向传播
                        optimizer.step()  # 优化权重
                        # TODO:在SummaryWriter中记录学习率
                        # ....

                # 计算损失值
                running_loss += loss.item() * inputs.size(0)  # loss计算的是平均值，所以要乘上batch-size，计算损失的总和
                running_corrects += (preds == labels).sum()  # 计算预测正确总个数
                # 每个batch加1次
                total_step[phase] += 1

            # 一轮训练完后计算损失率和正确率
            epoch_loss = running_loss / len(dataloaders[phase].sampler)  # 当前轮的总体平均损失值
            epoch_acc = float(running_corrects) / len(dataloaders[phase].sampler)  # 当前轮的总正确率

            accnp.append(epoch_acc)
            lossnp.append(epoch_loss)

            time_elapsed = time.time() - since
            print()
            print('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format(phase, epoch_loss, counter, epoch_acc))

            if phase == 'test':
                # 得到最好那次的模型
                if epoch_loss < test_loss_min:  # epoch_acc > best_acc:

                    best_acc = epoch_acc

                    # 保存当前模型
                    best_model_wts = copy.deepcopy(model_ft.state_dict())
                    state = {
                        'state_dict': model_ft.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                    }
                    # 只保存最近2次的训练结果
                    save_num = 0 if save_num > 1 else save_num
                    save_name_t = '{}_{}.pth'.format(filename, save_num)
                    torch.save(state, save_name_t)  # \033[1;31m 字体颜色：红色\033[0m
                    print("已保存最优模型，准确率:\033[1;31m {:.2f}%\033[0m，文件名：{}".format(best_acc * 100, save_name_t))
                    save_num += 1
                    test_loss_min = epoch_loss
                    counter = 0
                else:
                    counter += 1

        print()
        print('当前学习率 : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        print()

    accdf = pd.DataFrame(accnp)
    accdf.to_csv('06convnext-acc.csv')
    lossdf = pd.DataFrame(lossnp)
    lossdf.to_csv('06convnext-loss.csv')
    # 训练结束
    time_elapsed = time.time() - since
    print()
    print('任务完成！')
    print('任务完成总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('最高验证集准确率: {:4f}'.format(best_acc))
    save_num = save_num - 1
    save_num = save_num if save_num < 0 else 1
    save_name_t = '{}_{}.pth'.format(filename, save_num)
    print('最优模型保存在：{}'.format(save_name_t))
