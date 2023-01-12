import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y

'''
训练数据集可手动生成，设放大倍数为scale，考虑到原始数据未必会被scale整除，所以要重新规划一下图像尺寸，所以训练数据集的生成分为三步：
1.将原始图像通过双三次插值重设尺寸，使之可被scale整除，作为高分辨图像数据HR
2.将HR通过双三次插值压缩scale倍，为低分辨图像的原始数据
3.将低分辨图像通过双三次插值放大scale倍，与HR图像维度相等，作为低分辨图像数据LR
最后，可通过h5py将训练数据分块并打包
'''
# 生成训练集
def train(args):

    """
    def是python的关键字，用来定义函数。这里通过def定义名为train的函数，函数的参数为args,args这个参数通过外部命令行传入output
    的路径，通过h5py.File()方法的w模式--创建文件自己自写，已经存在的文件会被覆盖，文件的路径是通过args.output_path来传入
    """
    h5_file = h5py.File(args.output_path, 'w')
    #  #用于存储低分辨率和高分辨率的patch
    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        '''
        这部分代码的目的就是搜索指定文件夹下的文件并排序,for这一句包含了几个知识点：
        1.{}.format():-->格式化输出函数,从args.images_dir路径中格式化输出路径
        2.glob.glob():-->返回所有匹配的文件路径列表,将1得到的路径中的所有文件返回
        3.sorted():-->排序，将2得到的所有文件按照某种顺序返回，，默认是升序
        4.for x in *:   -->循换输出
        '''
        #将照片转换为RGB通道
        hr = pil_image.open(image_path).convert('RGB')
        '''
        1.  *.open(): 是PIL图像库的函数，用来从image_path中加载图像
        2.  *.convert(): 是PIL图像库的函数, 用来转换图像的模式
        '''
        #取放大倍数的倍数, width, height为可被scale整除的训练数据尺寸
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        #图像大小调整,得到高分辨率图像Hr
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        #低分辨率图像缩小
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        #低分辨率图像放大，得到低分辨率图像Lr
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        #转换为浮点并取ycrcb中的y通道
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)
        '''
        np.array():将列表list或元组tuple转换为ndarray数组
        astype():转换数组的数据类型
        convert_rgb_to_y():将图像从RGB格式转换为Y通道格式的图片
        假设原始输入图像为(321,481,3)-->依次为高，宽，通道数
        1.先把图像转为可放缩的scale大小的图片,之后hr的图像尺寸为(320,480,3)
        2.对hr图像进行双三次上采样放大操作
        3.将hr//scale进行双三次上采样放大操作之后×scale得到lr
        4.接着进行通道数转换和类型转换
        '''
        # 将数据分割
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                '''
                图像的shape是宽度、高度和通道数,shape[0]是指图像的高度=320;shape[1]是图像的宽度=480; shape[2]是指图像的通道数
                '''
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    #创建数据集,把得到的数据转化为数组类型
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()

#下同,生成测试集
def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--eval', action='store_true')  #store_flase就是存储一个bool值true，也就是说在该参数在被激活时它会输出store存储的值true。
    args = parser.parse_args()

    #决定使用哪个函数来生成h5文件，因为有俩个不同的函数train和eval生成对应的h5文件。
    if not args.eval:
        train(args)
    else:
        eval(args)
