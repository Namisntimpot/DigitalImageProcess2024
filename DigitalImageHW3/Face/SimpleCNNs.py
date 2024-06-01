import os
import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F

class SimpleCNN1(nn.Module):
    def __init__(self, n_feat=128):
        super().__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 第三层卷积层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 全连接层
        self.fc = nn.Linear(128 * 14 * 11, n_feat)
    
    def forward(self, x):
        # 第一层卷积 + ReLU + 池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 第二层卷积 + ReLU + 池化
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 第三层卷积 + ReLU + 池化
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 展平张量
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        return x
    

class SimpleCNN2(nn.Module):
    def __init__(self, n_feat=128):
        super().__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        
        # 第三层卷积层
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4)
        
        # 全连接层
        self.fc = nn.Linear(4 * 14 * 11, n_feat)
    
    def forward(self, x):
        # 第一层卷积 + ReLU + 池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 第二层卷积 + ReLU + 池化
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 第三层卷积 + ReLU + 池化
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 展平张量
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        return x
    

class SimpleCNN3(nn.Module):
    def __init__(self, n_feat=128):
        super().__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 第三层卷积层
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 全连接层
        self.fc = nn.Linear(64 * 14 * 11, n_feat)
    
    def forward(self, x):
        # 第一层卷积 + ReLU + 池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 第二层卷积 + ReLU + 池化
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 第三层卷积 + ReLU + 池化
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 展平张量
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        return x