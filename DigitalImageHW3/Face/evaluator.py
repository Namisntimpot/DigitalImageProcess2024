import os
import numpy as np

import torch

class Evaluator:
    def __init__(self, trained_model, data, label) -> None:
        self.data = data
        self.label = label
        self.representation = trained_model.get_all_representation(data)
        self.model = trained_model
    
    def evaluate(self, need_compare_matrix = False):
        '''
        评估误识率FAR（将非目标人认为是目标人的占比）和拒识率FRR（将目标人认为是非目标人的占比） \\
        返回：FAR, FRR, 正误矩阵（因为图片数量少，所以可以得到这么个矩阵）, 1是正确
        '''
        same_people_times = 0
        same_people_wrong = 0
        diff_people_times = 0
        diff_people_wrong = 0
        if need_compare_matrix:
            compare_matrix = np.zeros((self.label.shape[0], self.label.shape[0]))
            for i in range(self.label.shape[0]):
                compare_matrix[i,i] = 1
        for i in range(self.label.shape[0]):
            target_repre = self.representation[i]
            target_label = self.label[i]
            for j in range(self.label.shape[0]):
                if i==j:
                    continue
                ret = self.model.is_same_people(target_repre, self.representation[j])
                if target_label == self.label[j]:
                    same_people_times += 1
                    if not ret:  # 认为是不同的人
                        same_people_wrong += 1
                    elif need_compare_matrix:
                        compare_matrix[i, j] = 1                      
                else:
                    diff_people_times += 1
                    if ret:      # 认为是相同的人
                        diff_people_wrong += 1
                    elif need_compare_matrix:
                        compare_matrix[i, j] = 1
        
        FRR = same_people_wrong / same_people_times
        FAR = diff_people_wrong / diff_people_times
        if need_compare_matrix:
            return FAR, FRR, compare_matrix
        else:
            return FAR, FRR, None
