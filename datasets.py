import h5py   # 一个h5py文件是 “dataset” 和 “group” 二合一的容器。
import numpy as np
from torch.utils.data import Dataset

'''为这些数据创建一个读取类，以便torch中的DataLoader调用，而DataLoader中的内容则是Dataset，
    所以新建的读取类需要继承Dataset，并实现其__getitem__和__len__这两个成员方法。
'''

class TrainDataset(Dataset):  # 构建训练数据集，通过np.expand_dims将h5文件中的lr（低分辨率图像）和hr（高分辨率图像）组合为训练集
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx): #通过np.expand_dims方法得到组合的新数据
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):   #得到数据大小
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

# 与TrainDataset类似
class EvalDataset(Dataset):    # 构建测试数据集，通过np.expand_dims将h5文件中的lr（低分辨率图像）和hr（高分辨率图像）组合为验证集
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])