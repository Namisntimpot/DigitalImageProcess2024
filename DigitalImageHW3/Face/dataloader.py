import os
import numpy as np
import torch
import cv2
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

def load_orl_faces(dataset_path, flatten, to_tensor):
    '''
    加载orl人脸数据集，返回train, test人脸数据.  \\
    flatten: 把人脸图片压成一维.  \\
    to_tensor: 是否转化为torch.tensor.  \\
    返回：train_data, train_label, test_data, test_label
    '''
    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "test")
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # train.
    for sx in sorted(os.listdir(train_dir), key=lambda x: int(x[1:])):
        label = int(sx[1:]) - 1
        for i in range(1, 11):
            pgm = os.path.join(train_dir, sx, "{}.pgm".format(i))
            data = cv2.imread(pgm)[:,:,0]  # 直接读出灰度图是(112, 92, 3)
            if flatten:
                data = data.reshape((data.shape[0] * data.shape[1]))
            if to_tensor:
                data = data[np.newaxis, : ,:]
            train_data.append(data)
            train_label.append(label)
    # test
    for sx in sorted(os.listdir(test_dir),key=lambda x: int(x[1:])):
        label = int(sx[1:]) - 1
        for i in range(1, 11):
            pgm = os.path.join(test_dir, sx, "{}.pgm".format(i))
            data = cv2.imread(pgm)[:,:,0]  # 直接读出灰度图是(112, 92, 3)
            if flatten:
                data = data.reshape((data.shape[0] * data.shape[1]))
            if to_tensor:
                data = data[np.newaxis, : ,:]
            test_data.append(data)
            test_label.append(label)
    # 转化为numpy
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    if to_tensor:
        train_data = torch.from_numpy(train_data).to(torch.float32)
        train_label = torch.from_numpy(train_label)
        test_data = torch.from_numpy(test_data).to(torch.float32)
        test_label = torch.from_numpy(test_label)
    return train_data, train_label, test_data, test_label


class FaceDataset(Dataset):
    def __init__(self, data:torch.Tensor, label:torch.Tensor) -> None:
        '''
        data, label需要已经提前变为tensor，提前放到gpu中.
        '''
        self.data = data
        self.label = label
        super().__init__()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]


if __name__=='__main__':
    load_orl_faces("orl_faces", 1, 0)