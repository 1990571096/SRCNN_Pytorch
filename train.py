import argparse
import os
import copy

import numpy as np
from torch import Tensor
import torch
from torch import nn
import torch.optim as optim

# gpu加速库
import torch.backends.cudnn as cudnn

from torch.utils.data.dataloader import DataLoader

# 进度条
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

##需要修改的参数
# epoch.pth
# losslog
# psnrlog
# best.pth

'''
python train.py --train-file "path_to_train_file" \
                --eval-file "path_to_eval_file" \
                --outputs-dir "path_to_outputs_file" \
                --scale 3 \
                --lr 1e-4 \
                --batch-size 16 \
                --num-epochs 400 \
                --num-workers 0 \
                --seed 123  
'''
if __name__ == '__main__':

    # 初始参数设定
    parser = argparse.ArgumentParser()   # argparse是python用于解析命令行参数和选项的标准模块
    parser.add_argument('--train-file', type=str, required=True,)  # 训练 h5文件目录
    parser.add_argument('--eval-file', type=str, required=True)  # 测试 h5文件目录
    parser.add_argument('--outputs-dir', type=str, required=True)   #模型 .pth保存目录
    parser.add_argument('--scale', type=int, default=3)  # 放大倍数
    parser.add_argument('--lr', type=float, default=1e-4)   #学习率
    parser.add_argument('--batch-size', type=int, default=16) # 一次处理的图片大小
    parser.add_argument('--num-workers', type=int, default=0)  # 线程数
    parser.add_argument('--num-epochs', type=int, default=400)  #训练次数
    parser.add_argument('--seed', type=int, default=123) # 随机种子
    args = parser.parse_args()

    # 输出放入固定文件夹里
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    # 没有该文件夹就新建一个文件夹
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    # benckmark模式，加速计算，但寻找最优配置，计算的前馈结果会有差异
    cudnn.benchmark = True

    # gpu或者cpu模式，取决于当前cpu是否可用
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 每次程序运行生成的随机数固定
    torch.manual_seed(args.seed)

    # 构建SRCNN模型，并且放到device上训练
    model = SRCNN().to(device)

    # 恢复训练，从之前结束的那个地方开始
    # model.load_state_dict(torch.load('outputs/x3/epoch_173.pth'))

    # 设置损失函数为MSE
    criterion = nn.MSELoss()

    # 优化函数Adam，lr代表学习率，
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    # 预处理训练集
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(
        # 数据
        dataset=train_dataset,
        # 分块
        batch_size=args.batch_size,
        # 数据集数据洗牌,打乱后取batch
        shuffle=True,
        # 工作进程，像是虚拟存储器中的页表机制
        num_workers=args.num_workers,
        # 锁页内存，不换出内存，生成的Tensor数据是属于内存中的锁页内存区
        pin_memory=True,
        # 不取余，丢弃不足batchSize大小的图像
        drop_last=True)
    # 预处理验证集
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # 拷贝权重
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    # 画图用
    lossLog = []
    psnrLog = []

    # 恢复训练
    # for epoch in range(args.num_epochs):
    for epoch in range(1, args.num_epochs + 1):
        # for epoch in range(174, 400):
        # 模型训练入口
        model.train()

        # 变量更新，计算epoch平均损失
        epoch_losses = AverageMeter()

        # 进度条，就是不要不足batchsize的部分
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            # t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs - 1))
            t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs))

            # 每个batch计算一次
            for data in train_dataloader:
                # 对应datastes.py中的__getItem__，分别为lr,hr图像
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)
                # 送入模型训练
                preds = model(inputs)

                # 获得损失
                loss = criterion(preds, labels)

                # 显示损失值与长度
                epoch_losses.update(loss.item(), len(inputs))

                # 梯度清零
                optimizer.zero_grad()

                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

                # 进度条更新
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        # 记录lossLog 方面画图
        lossLog.append(np.array(epoch_losses.avg))
        # 可以在前面加上路径
        np.savetxt("lossLog.txt", lossLog)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        # 是否更新当前最好参数
        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 验证不用求导
            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        # 记录psnr
        psnrLog.append(Tensor.cpu(epoch_psnr.avg))
        np.savetxt('psnrLog.txt', psnrLog)
        # 找到更好的权重参数，更新
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))

        torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))

    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
