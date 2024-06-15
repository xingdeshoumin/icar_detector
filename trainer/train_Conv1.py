#  导入包和定义参数
import sys
import pynvml
import torch
import torch.utils.data as tud
import torch.nn as nn
from tqdm import tqdm
import argparse
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('MLP')
    parser.add_argument('--flip', type=bool, default=False)
    parser.add_argument('--grid', type=bool, default=True)
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--epochs', type=int, help='the number of epoch', default=160)
    parser.add_argument('--batch_size', type=int, default=40, help='input batch size for training (default: 15)')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=int, default=1.0)
    parser.add_argument("--dataset_path", type=str,
                        default='./Sign_dataset/')
    parser.add_argument("--save_path", type=str, help="the path of model saved",
                        default='./Sign_models/')
    args = parser.parse_args()  # 也可直接使用 args, _ = parser.parse_known_args()
    return args

#  定义网络和模型 三层的网络
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2),
            nn.Flatten(),
            nn.Linear(28 * 28 * 8, 164),
            nn.Linear(164, 84),
            # nn.Dropout(p=0.5),
            nn.Linear(84, 7)
        )

    def forward(self, x):
        x = self.model(x)
        return x

def main(args):

    best_acc = 0.0
    epoch_list = [120, 140, 160, 180, 200, 220, 240, 260, 280]
    lr_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    best_acc_list = []
    cont_i = 0
    cont_j = 0

    while (1):

        # 初始化NVML库
        pynvml.nvmlInit()
        # 获取第一个GPU设备的句柄
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if (args.log):
            print("using {} device.".format(device))

        # 定义随机仿射变换
        affine_transform = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2))
        # 定义一个变换列表
        transform_list = [
            affine_transform,
            transforms.Resize([28, 28]),  # 裁剪图片
            transforms.Grayscale(1),  # 单通道
            transforms.ToTensor(),  # 将图片数据转成tensor格式
        ]

        # 根据变量来决定是否添加RandomHorizontalFlip变换
        if args.flip:
            transform_list.insert(0, transforms.RandomHorizontalFlip())  # 在列表的开头添加RandomHorizontalFlip变换

        # 使用Compose将变换列表转换为一个组合变换
        transform = transforms.Compose(transform_list)

        # 使用ImageFolder加载数据集
        torch_dataset = datasets.ImageFolder(args.dataset_path, transform=transform)

        train_size, val_size = int(len(torch_dataset)*args.ratio), len(torch_dataset) - int(len(torch_dataset)*args.ratio)
        train_dataset, val_dataset = tud.random_split(torch_dataset, [train_size, val_size])
        if args.grid:
            batch_size = epoch_list[cont_i]
        else:
            batch_size = args.batch_size
        train_loader = tud.DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True, )
        val_loader = tud.DataLoader(dataset=val_dataset,
                                    batch_size=batch_size,
                                    shuffle=True, )
        if (args.log):
            print("using {}  for training, {}   for validation.".format(train_size, val_size))

        net = MLP()
        net.to(device)

        # 损失函数
        loss = nn.CrossEntropyLoss()
        loss = loss.to(device)  # 调用GPU
        # 优化器
        if args.grid:
            lr = lr_list[cont_j]
        else:
            lr = args.lr
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

        if args.grid:
            epochs = epoch_list[cont_i]
        else:
            epochs = args.epochs

        for epoch in range(epochs):
            # train
            net.train()
            if args.grid:
                train_bar = train_loader
            else:
                train_bar = tqdm(train_loader, file=sys.stdout)
            for step, train_data in enumerate(train_bar):
                imgs, targets = train_data
                imgs = imgs.to(device)  # 调用GPU
                targets = targets.to(device)
                output = net(imgs)

                # 计算损失
                result_loss = loss(output, targets)

                optimizer.zero_grad()
                result_loss.backward()
                optimizer.step()

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                if args.grid:
                    val_bar = val_loader
                else:
                    val_bar = tqdm(val_loader, file=sys.stdout)
                for val_bar in val_bar:
                    imgs, targets = val_bar
                    imgs = imgs.to(device)
                    targets = targets.to(device)

                    output = net(imgs)
                    result_loss = loss(output, targets)

                    predict_y = torch.max(output, dim=1)[1]
                    acc += torch.eq(predict_y, targets).sum().item()

            val_accurate = acc / val_size
            if (args.log):
                print(f'[epoch {epoch+1}] , best_acc: {best_acc:.3f}, now_acc: {val_accurate:.3f}')
            # 获取GPU内存信息
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if (args.log):
                print(f'GPU内存占用 : {(info.used/info.total):.3f}')
        if val_accurate > best_acc:
            best_acc = val_accurate
            filename = f"{args.save_path}model_{best_acc:.3f}percent_Conv.pth"
            torch.save(net.state_dict(), filename)
            filename = f"{args.save_path}model_{best_acc:.3f}percent_Conv.onnx"
            dummy_input = torch.randn(batch_size, 1, 28, 28, device=device)
            torch.onnx.export(net, dummy_input, filename)
            print(f'best_acc: {best_acc:.3f} \nepoch: {epoch_list[cont_i]} \nlr: {lr_list[cont_j]}\n------------------\
                        ---------------------------------------------------')

        if args.grid:
            cont_i +=1
            if (cont_i >= len(epoch_list)):
                cont_i = 0
                cont_j +=1
            if (cont_j >= len(lr_list)):
                break
        if (args.log):
            print('Finished Training')


if __name__ == '__main__':
    args_ = parse_option()
    main(args_)
