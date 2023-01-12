import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


if __name__ == '__main__':
    # 设置权重参数目录，处理图像目录，放大倍数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', default='outputs/x3/best.pth', type=str)
    parser.add_argument('--image-file', default='img/img_kenan.jpeg', type=str)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()
    #  Benchmark模式会提升计算速度
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)   # 新建一个模型

    state_dict = model.state_dict()  # 通过 model.state_dict()得到模型有哪些 parameters and persistent buffers
    # torch.load('tensors.pth', map_location=lambda storage, loc: storage)  使用函数将所有张量加载到CPU（适用在GPU训练的模型在CPU上加载）
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():   # 载入最好的模型参数
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()   # 切换为测试模式 ，取消dropout

    image = pil_image.open(args.image_file).convert('RGB')   # 将图片转为RGB类型

    # 经过一个插值操作，首先将原始图片重设尺寸，使之可以被放大倍数scale整除
    # 得到低分辨率图像Lr，即三次插值后的图像，同时保存输出
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
    # 将图像转化为数组类型，同时图像转为ycbcr类型
    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)
    # 得到 ycbcr中的 y 通道
    y = ycbcr[..., 0]
    y /= 255.  # 归一化处理
    y = torch.from_numpy(y).to(device) #把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变，并且将参数放到device上
    y = y.unsqueeze(0).unsqueeze(0)  # 增加两个维度
    # 令reqires_grad自动设为False，关闭自动求导
    # clamp将inputs归一化为0到1区间
    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(y, preds)   # 计算y通道的psnr值
    print('PSNR: {:.2f}'.format(psnr))  # 格式化输出PSNR值

    # 1.mul函数类似矩阵.*，即每个元素×255
    # 2. *.cpu（）.numpy（） 将数据的处理设备从其他设备（如gpu拿到cpu上），不会改变变量类型，转换后仍然是Tensor变量，同时将Tensor转化为ndarray
    # 3. *.squeeze(0).squeeze(0)数据的维度进行压缩
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)  #得到的是经过模型处理，取值在[0,255]的y通道图像

    # 将img的数据格式由（channels,imagesize,imagesize）转化为（imagesize,imagesize,channels）,进行格式的转换后方可进行显示。
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])

    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)  # 将图像格式从ycbcr转为rgb，限制取值范围[0,255]，同时矩阵元素类型为uint8类型
    output = pil_image.fromarray(output)   # array转换成image，即将矩阵转为图像
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))  # 对图像进行保存
