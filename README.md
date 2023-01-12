# SRCNN_Pytorch
 CSDN博客代码讲解地址：https://blog.csdn.net/weixin_52261094/article/details/128389448


## Requirements
- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0


### Step1: 训练集，测速集下载地址，
#### 也可以用 preapre自己制作训练集和测试集

| Dataset | Scale | Type | Link |
  |---------|-------|------|------|
  | 91-image | 2 | Train | [Download](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/91-image_x2.h5) |
  | 91-image | 3 | Train | [Download](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/91-image_x3.h5) |
  | 91-image | 4 | Train | [Download](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/91-image_x4.h5) |
  | Set5 | 2 | Eval | [Download](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/Set5_x2.h5) |
  | Set5 | 3 | Eval | [Download](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/Set5_x3.h5) |
  | Set5 | 4 | Eval | [Download](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/Set5_x4.h5) |

  Download any one of 91-image and Set5 in the same Scale and then **move them under `./datasets` as `./datasets/91-image_x2.h5` and `./datasets/Set5_x2.h5`**

### Step2: 训练模型

    --train-file "path_to_train_file" \
    --eval-file "path_to_eval_file" \
    --outputs-dir "path_to_outputs_file" \
    --scale 3 \
    --lr 1e-4 \
    --batch-size 16 \
    --num-epochs 400 \
    --num-workers 0 \
    --seed 123     




### Step3: 400轮训练结果，训练得到的最优权重

- [trained by x3](https://pan.baidu.com/s/1sLGsDPuC7BCUaVMDRv013A?pwd=1234)

**将权重移动到该目录文件下： `./outputs` as `./outputs/x3/best.pth`**


