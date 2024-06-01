import os
import numpy as np
import cv2
from sklearn.decomposition import PCA

from dataloader import *
from evaluator import *

class PCA_Face:
    def __init__(self, n_components, thresh = None) -> None:
        '''
        需要把特征降到几维
        '''
        self.pca = PCA(n_components)
        self.u = None  # 平均
        self.thresh = thresh
        
    def fit(self, X, label):
        self.u = np.mean(X, axis=0)  # (10340,)
        print(self.u.shape)
        x_minus_u = X - self.u
        self.pca.fit(x_minus_u)
        trans_x = self.pca.transform(X)
        trans_x_norm = np.linalg.norm(trans_x, axis=1)
        # 寻找一个阈值，取所有训练类中最大的余弦距离. 这里写死，每类固定有10张照片
        if self.thresh is None:
            n_subjects = int(label.shape[0] / 10)  # 有这么多个人
            self.thresh = -1
            for k in range(n_subjects):
                for i in range(10):
                    i_ind = k * 10 + i
                    for j in range(10):
                        j_ind = k*10+j
                        dis = np.arccos(np.dot(trans_x[i_ind, :], trans_x[j_ind, :]) / (trans_x_norm[i_ind] * trans_x_norm[j_ind])) / np.pi
                        if self.thresh < dis:
                            self.thresh = dis
        print("threshold: {}".format(self.thresh))

    def predict(self, Y):
        return self.pca.transform(Y - self.u)
    
    def get_all_representation(self ,Y):
        return self.pca.transform(Y - self.u)
                    

    def disfunc(self, X, Y):
        '''
        比较两个一维特征间的距离.
        '''
        # L2范数
        X_norm = np.linalg.norm(X)
        Y_norm = np.linalg.norm(Y)
        cosine_sim = np.dot(X, Y) / (X_norm * Y_norm)
        return np.arccos(cosine_sim) / np.pi
    
    def is_same_people(self, X, Y):
        '''
        X, Y需要是representation.
        '''
        dis = self.disfunc(X, Y)
        return dis < self.thresh


if __name__=='__main__':
    train_data, train_label, test_data, test_label = load_orl_faces("orl_faces", True, False)
    print(train_label)
    pca_face = PCA_Face(128, 0.235)
    pca_face.fit(train_data, train_label)
    # trans_x = pca_face.predict(train_data)  # threshold 设置为0.39
    # for i in range(trans_x.shape[0]):
    #     print(pca_face.disfunc(trans_x[0], trans_x[i]))
    evaluat = Evaluator(pca_face, test_data, test_label)
    FAR, FRR, _ = evaluat.evaluate()
    print(FAR, FRR)   # 约等于乱猜