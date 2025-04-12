def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
import plotly.express as px
import os
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import time

from hyperactive import Hyperactive
from hyperactive.optimizers import SimulatedAnnealingOptimizer
from hyperactive.optimizers import PowellsMethod

import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator

from fitness import fitness_function
from models import demomodel

color_scale = px.colors.sequential.Jet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 32
DIR = os.getcwd()
EPOCHS = 100
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 100
N_VALID_EXAMPLES = BATCHSIZE * 40

transform_train = transforms.Compose([
    transforms.ToTensor()
    , transforms.Resize((112, 112))
    # , transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    , transforms.RandomCrop(112, padding=4)  # 先四周填充0，在吧图像随机裁剪成32*32
    , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
    , transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 均值，标准差
])
transform_valid = transforms.Compose([
    transforms.ToTensor()
    , transforms.Resize((112, 112))
    # , transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    , transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# Get the MNIST dataset.
data_flag = 'bloodmnist'
download = True
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
CLASSES = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', transform=transform_train, download=download)
# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, shuffle=True)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, shuffle=True)

valid_dataset = DataClass(split='val', transform=transform_valid, download=download)
valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=BATCHSIZE, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=BATCHSIZE, shuffle=False)


def Cost_value(acc, params, use_time):
    Cost_value = fitness_function(accuracy_increase=acc, params=params, training_time=use_time)
    return Cost_value


def pytorch_cnn(params):

    architecture_params = {
        "stage1": {
            "dim": params["dims_1"],
            "depth": params["layer_1"],
            "mlp_ratio": params["mlp_ratio_1"]
        },
        "stage2": {
            "dim": params.get("stage2_dims", 0),  # 使用get避免KeyError
            "depth": params.get("stage2_layers", 0),
            "mlp_ratio": params.get("stage2_mlp_ratio", 3)
        },
        "block_type": params["block_type"],
        "act_layer": params["activation"],
        "use_se": params["use_se"],
        "kernel_size": params["kernel_size"]
    }

    demo_model = demomodel.demomodel_any(params=architecture_params, num_classes=CLASSES)
    # 计算的参数量
    total_params_o = sum(p.numel() for p in demo_model.parameters())

    optimizer = getattr(optim, "Adam")(demo_model.parameters(), lr=0.01)
    # print(demo_model)
    # Training of the model.
    start_time: float = time.time()

    criterion = nn.CrossEntropyLoss()

    since = time.time()
    for epoch in range(EPOCHS):
        sum_loss = 0
        demo_model.to(DEVICE)
        demo_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(DEVICE), target.to(DEVICE)
            targets = target.squeeze(dim=1).long()

            optimizer.zero_grad()
            output = demo_model(data)
            loss = criterion(output, targets)
            loss.to(DEVICE)
            loss.backward()
            optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            ave_loss = sum_loss / len(train_loader)
        # Validation of the model.

        demo_model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                targetss = target.squeeze(dim=1).long()
                output = demo_model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(targetss.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

    end_time = time.time()
    use_time = end_time - start_time
    cost_value = Cost_value(accuracy, total_params_o, use_time)
    time_elapsed = time.time() - since
    print()
    print('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Loss: {:.4f}, Acc: {:.4f}'.format(ave_loss, accuracy))

    return cost_value


search_space = {
    "dims_1": list(range(16, 128, 8)),
    "layer_1": list(range(2, 12, 2)),
    "mlp_ratio_1": [1, 2, 3, 4, 5],

    "block_type": ["conv", "transformer","DW"],
    "activation": ["relu", "gelu", "swish"],
    "use_se": [True, False],
    "kernel_size": [3, 5, 7],
    "expansion_ratio": [2, 3, 4],

    "stage2_dims": list(range(32, 96, 16)),
    "stage2_layers": list(range(1, 5)),
    "stage2_mlp_ratio": [2, 3, 4],
}

if __name__ == '__main__':
    c_time1 = time.time()
    hyper = Hyperactive(verbosity=False)
    # 设定全局优化方法
    optimizer = PowellsMethod(iters_p_dim=20, )
    # 开始全局优化搜索
    hyper.add_search(pytorch_cnn, search_space, optimizer=optimizer, n_iter=20)
    hyper.run()

    search_data = hyper.search_data(pytorch_cnn, times=True)
    search_data.to_csv("Powell_value(Fastvit).csv")

    best_para = hyper.best_para(pytorch_cnn)
    print(best_para)
    d_time1 = time.time() - c_time1
    print("Optimization time 1:", round(d_time1, 2))

    # 首次位置的FataFrame传入，进行二次训练时，避免位置的重新训练 search_data
    c_time2 = time.time()
    # 二次搜索，传入初次搜索的最佳位置，最佳位置处开始搜索
    initialize = {"random": 4, "warm_start": [best_para]}

    search_space1 = {
        "dims_1": list(range(64, 128, 16)),
        "layer_1": list(range(6, 12, 2)),
        "mlp_ratio_1": [3, 4, 5],

        "block_type": ["transformer"],
        "activation": ["gelu"],
        "use_se": [True],
        "kernel_size": [5, 7],

        "stage2_dims": list(range(64, 96, 16)),
        "stage2_layers": list(range(2, 4)),
        "stage2_mlp_ratio": [3, 4],
        "block_type": ["conv", "transformer"],
        "activation": ["relu", "gelu", "swish"],
    }

    hyper1 = Hyperactive(verbosity=False)
    optimizer1 = SimulatedAnnealingOptimizer(
        epsilon=0.1,
        distribution="laplace",
        n_neighbours=4,
        rand_rest_p=0.1,
        p_accept=0.15,
        norm_factor="adaptive",
        annealing_rate=0.999,
        start_temp=0.8,
    )
    hyper1.add_search(pytorch_cnn,
                      search_space1,
                      n_iter=20,
                      n_jobs=1,
                      optimizer=optimizer1,
                      memory_warm_start=search_data,
                      initialize=initialize)
    hyper1.run()

    d_time2 = time.time() - c_time2
    print("Optimization time 2:", round(d_time2, 2))
    print("\n The second optimization run is " + '{}'.format(
        round((1 - d_time2 / d_time1) * 100, 2)) + "% faster than the first one.")

    search_data1 = hyper1.search_data(pytorch_cnn, times=True)

    # merge the search data from the previous run and the current run合并前一次运行和当前运行的搜索数据
    search_data1_ = search_data1.append(search_data, ignore_index=True)
    search_data1_
    search_data1_.to_csv("final_value(Fastvit).csv")

    # 绘制搜索时间缩短的图像
    search_data_1 = hyper1.search_data(pytorch_cnn, times=True)
    search_data = search_data_1.append(search_data, ignore_index=True)

    # times in seconds
    eval_times = search_data["eval_times"]
    eval_times_mem = search_data_1["eval_times"]

    opt_times = search_data["iter_times"] - search_data["eval_times"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=eval_times, name="evaluation time", nbinsx=15))
    fig.add_trace(go.Histogram(x=eval_times_mem, name="evaluation time second run", nbinsx=15))
    fig.add_trace(go.Histogram(x=opt_times, name="optimization time", nbinsx=15))
    fig.show()
