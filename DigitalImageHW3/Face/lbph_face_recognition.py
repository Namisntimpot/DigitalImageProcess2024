import os
import numpy as np
import cv2
from dataloader import *
from evaluator import *
#from skimage.feature import local_binary_pattern

class LBPH_Face:
    def __init__(self, thresh):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.thresh = thresh

    def fit(self, X, label):
        self.recognizer.train(X, label)

    def extract_lbph_features(self, image, grid_x=8, grid_y=8, num_points=24, radius=8):
        """
        计算图像的LBPH特征向量
        :param image: 灰度图像
        :param grid_x: 网格的列数
        :param grid_y: 网格的行数
        :param num_points: LBP算法中的点数
        :param radius: LBP算法中的半径
        :return: LBPH特征向量
        """
        lbp_hist = []

        # 获取图像的尺寸
        height, width = image.shape

        # 计算网格的大小
        cell_w = width // grid_x
        cell_h = height // grid_y

        # 对图像应用LBP
        #lbp = local_binary_pattern(image, num_points, radius)
        lbp = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
        for i in range(radius, height + radius):
            for j in range(radius, width + radius):
                center = lbp[i, j]
                binary = [1 if lbp[i + k, j + m] > center else 0 for k, m in zip([1, 0, -1, -1, -1, 0, 1, 1], [0, 1, 1, 0, -1, -1, -1, 0])]
                binary_str = ''.join(map(str, binary))
                lbp[i - radius, j - radius] = int(binary_str, 2)

        # 遍历每个网格单元，计算直方图
        for i in range(grid_y):
            for j in range(grid_x):
                cell = lbp[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                hist = cv2.calcHist([cell], [0], None, [256], [0, 256])
                lbp_hist.extend(hist.flatten())

        return np.array(lbp_hist)
    
    def get_all_representation(self, data):
        ret = []
        for i in range(data.shape[0]):
            ret.append(self.extract_lbph_features(data[i]))
        return np.array(ret)
    
    def is_same_people(self, repreA, repreB):
        cos_sim = np.dot(repreA, repreB) / (np.linalg.norm(repreA) * np.linalg.norm(repreB))
        return cos_sim > self.thresh
    

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_orl_faces("./orl_faces", False, False)
    lbph = LBPH_Face(0.75)
    lbph.fit(train_data, train_label)
    eva = Evaluator(lbph, test_data, test_label)
    FAR, FRR, _ = eva.evaluate(False)
    print(FAR, FRR)