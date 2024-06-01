import os
import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from SimpleCNNs import *


class ArcFace(nn.Module):
    def __init__(self, dim_feature, num_classes, s=64, m=0.1) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.FloatTensor(dim_feature, num_classes))
        nn.init.xavier_uniform_(self.weights)

    
    def forward(self, x):
        '''
        x: 输入的上一层的特征，(n_batch, n_feature)  \\
        起到相当于softmax层的作用
        '''
        x_norm = F.normalize(x, 2, 1)
        weights_norm = F.normalize(self.weights, 2, 0)  # 在feature那个维度算L2范数
        cos = torch.clamp(torch.matmul(x_norm, weights_norm), 0, 1)   # 要防止这个最终结果超过1，超出cos值的范围 (n_batches, n_classes)
        acos = torch.acos(cos)  # theta角度.
        numerator = torch.exp(self.s * torch.cos(acos + self.m))
        denominator = torch.sum(torch.exp(self.s * cos), dim=1, keepdim=True) - torch.exp(self.s * cos)  # 沿着n_classes那一维相加. keepdim保持shape不变.

        return torch.log(numerator / (numerator + denominator))  # (n_batches, n_classes)

    

class FaceNet(nn.Module):
    def __init__(self, feature_extractor:nn.Module, classification_layer:nn.Module, training_mode, thresh=0.75) -> None:
        super().__init__()
        self.feat_extractor = feature_extractor
        if training_mode:
            self.outputlayer = classification_layer
        self.training_mode = training_mode
        self.thresh = thresh
        
    def forward(self, x):
        '''
        如果是训练模式，返回feature (n_batches, n_feat), classes(n_batches, n_classes) \\
        否额只返回feature. \\
        feature是经过L2正则化的
        '''
        feat = F.normalize(self.feat_extractor(x), 2, 1)
        if self.training_mode:
            classes = self.outputlayer(feat)
            return feat, classes
        return feat
    

    def NllLoss(self, classes, label):
        '''
        classes: (n_batches, n_classes)
        label: (n_batches)  必须是tensor.
        '''
        assert self.training_mode
        return F.nll_loss(classes, label.long())
    
    def get_all_representation(self, data:torch.Tensor):
        assert not self.training_mode
        n_samples = data.size(0)
        n_batch = 8
        i = 0
        representation = []
        while i + n_batch <= n_samples:
            input_x = data[i:i+n_batch,...]
            output = self.forward(input_x).detach().cpu().numpy()
            representation.append(output)
            i += n_batch
        if i < n_samples:
            input_x = data[i:n_samples,...]
            output = self.forward(input_x).detach().cpu().numpy()
            representation.append(output)
        return np.concatenate(representation, 0)
    
    def is_same_people(self, repreA:np.ndarray, repreB:np.ndarray):
        # 已经经过L2正则化了
        # 使用余弦相似度
        A_l2 = np.linalg.norm(repreA)
        B_l2 = np.linalg.norm(repreB)
        cos_sim = np.dot(repreA, repreB) / (A_l2 * B_l2)
        return cos_sim > self.thresh
        # 欧几里得距离
        # dis = np.sum((repreA - repreB)**2) / repreB.shape[0]
        # print(1-dis)
        # return 1-dis > self.thresh
    
    def get_feat_extractor_state_dict(self):
        return self.feat_extractor.state_dict()
    
    def load_feat_extractor_state_dict(self, state_dict):
        self.feat_extractor.load_state_dict(state_dict)
    
    def get_arcface_state_dict(self):
        return self.outputlayer.state_dict()
    

class FaceNet_ResNet18(FaceNet):
    def __init__(self,dim_feat, num_classes, s = 64, m = 0.1, training_mode = False) -> None:
        feat_extractor = resnet18(pretrained=False)
        feat_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feat_extractor.fc = nn.Linear(512, dim_feat)
        if training_mode:
            outputlayer = ArcFace(dim_feat, num_classes, s, m)
            super().__init__(feat_extractor, outputlayer, training_mode)
        else:
            super().__init__(feat_extractor, None, training_mode)


class FaceNet_ResNet18_Pretrained(FaceNet):
    def __init__(self):
        feat_extractor = resnet18(pretrained=True)
        super().__init__(feat_extractor, None, False)

    def forward(self, x):
        x = torch.cat([x]*3, dim=1)
        return super().forward(x)


class FaceNet_SimpleCNN1(FaceNet):
    def __init__(self,dim_feat, num_classes, s = 64, m = 0.1, training_mode = False) -> None:
        feat_extractor = SimpleCNN1(dim_feat)
        if training_mode:
            outputlayer = ArcFace(dim_feat, num_classes, s, m)
            super().__init__(feat_extractor, outputlayer, training_mode)
        else:
            super().__init__(feat_extractor, None, training_mode)


class FaceNet_SimpleCNN2(FaceNet):
    def __init__(self,dim_feat, num_classes, s = 64, m = 0.1, training_mode = False) -> None:
        feat_extractor = SimpleCNN2(dim_feat)
        if training_mode:
            outputlayer = ArcFace(dim_feat, num_classes, s, m)
            super().__init__(feat_extractor, outputlayer, training_mode)
        else:
            super().__init__(feat_extractor, None, training_mode)

class FaceNet_SimpleCNN3(FaceNet):
    def __init__(self,dim_feat, num_classes, s = 64, m = 0.1, training_mode = False) -> None:
        feat_extractor = SimpleCNN3(dim_feat)
        if training_mode:
            outputlayer = ArcFace(dim_feat, num_classes, s, m)
            super().__init__(feat_extractor, outputlayer, training_mode)
        else:
            super().__init__(feat_extractor, None, training_mode)

if __name__=='__main__':
    img = cv2.imread("./orl_faces/train/s1/1.pgm")[:,:,0][np.newaxis, np.newaxis, :, :]
    img = torch.from_numpy(img).to(torch.float32)
    facenet = FaceNet_ResNet18(128, 24, training_mode=True)
    feat, cla = facenet.forward(img)
    print(feat.shape)
    print(cla.shape)