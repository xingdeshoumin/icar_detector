#  导入包和定义参数
import sys
import torch
import torch.utils.data as tud
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import argparse
from network import MLP
from torchvision import datasets, transforms


def parse_option():
    parser = argparse.ArgumentParser('MLP')
    parser.add_argument('--epochs', type=int, help='the number of epoch', default=20)
    parser.add_argument('--batch_size', type=int, default=20, help='input batch size for training (default: 15)')
    parser.add_argument('--lr', type=int, default=0.1)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=int, default=1.0)
    parser.add_argument("--save_path", type=str, help="the path of model saved",
                        default='../../models/')
    args = parser.parse_args()  # 也可直接使用 args, _ = parser.parse_known_args()
    return args

def main(args):
    best_acc = 0.0
    while (1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # 创建转换器
        transform = transforms.Compose([
            transforms.Resize([28, 28]),  # 裁剪图片
            transforms.Grayscale(1),  # 单通道
            transforms.ToTensor(),  # 将图片数据转成tensor格式
        ])

        # 使用ImageFolder加载数据集
        torch_dataset = datasets.ImageFolder('D:/Miniconda3/envs/ML_01/Data/MLP/deep_learning_for_cv/1_classification/c02_MLP/dataset/', transform=transform)

        train_size, val_size = int(len(torch_dataset)*0.8), len(torch_dataset) - int(len(torch_dataset)*0.8)
        train_dataset, val_dataset = tud.random_split(torch_dataset, [train_size, val_size])
        train_loader = tud.DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True, )
        val_loader = tud.DataLoader(dataset=val_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True, )
        print("using {}  for training, {}   for validation.".format(train_size, val_size))

        net = MLP()
        net.to(device)

        # pata = list(net.parameters())
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)  # 设置学习率下降策略
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            # train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, train_data in enumerate(train_bar):
                train_x, labels = train_data
                # zero the parameter gradients
                optimizer.zero_grad()
                # net.zero_grad()
                # forward + backward + optimize
                # output = net(train_x.to(device))
                train_x = train_x.to(device)
                output = net.forward(train_x.view(-1, 28 * 28))
                loss = criterion(output, labels.to(device))
                # loss = torch.nn.functional.nll_loss(output, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if step % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'train_data:[{epoch + 1}, {step + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

            scheduler.step()  # 更新学习率
            print(f'[last_lr {scheduler.get_last_lr()}]')

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout)
                for val_bar in val_bar:
                    val_x, val_labels = val_bar
                    # outputs = net(val_x.to(device))
                    val_x = val_x.to(device)
                    output = net.forward(val_x.view(-1, 28 * 28))
                    predict_y = torch.max(output, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / val_size
            print(f'[epoch {epoch+1}] , val_accuracy: {val_accurate:.3f}')
            if val_accurate > best_acc:
                best_acc = val_accurate
                filename = f"{args.save_path}model_{best_acc}percent.pth"
                torch.save(net.state_dict(), filename)
                filename = f"{args.save_path}model_{best_acc}percent.onnx"
                dummy_input = torch.randn(args.batch_size, 1, 784, device=device)
                torch.onnx.export(net, dummy_input, filename)
        print('Finished Training')


if __name__ == '__main__':
    args_ = parse_option()
    main(args_)
