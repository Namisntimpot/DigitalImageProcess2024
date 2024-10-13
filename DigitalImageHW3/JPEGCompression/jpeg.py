import os
import sys
import numpy as np
import cv2
from jpeg_compress import jpeg_compress
from jpeg_output import *
from bit_outstream import *
from utils import *

def jpeg(img:np.ndarray, out_path:str, quality_coef = 1):
    '''
    img: YCbCr格式的图片.
    '''
    f = BitOutStream(out_path)
    h, w, _ = img.shape
    compressed = jpeg_compress(img, quality_coef)
    # 输出SOI文件头.
    output_soi(f)
    # 输出APPO
    output_app0(f)
    # 定义量化表DQT, 
    output_dqt(f, np.uint(QuanTable.Lum / quality_coef), 0)  # 亮度的量化表号是0.
    output_dqt(f, np.uint(QuanTable.Chrom / quality_coef), 1)# 色度的量化表号是1.
    # 定义图像基本信息SOF0
    output_sof0(f, h, w, 0, 1)
    # 定义四个哈夫曼表. no.0: Y的DC(type=0)表, no.1: Y的AC(type=1)表
    #                 no.2: C的DC表，no.3: C的AC表
    output_dht(f, 0, 0, compressed['huffman_tup_dc_y'])
    output_dht(f, 1, 1, compressed['huffman_tup_ac_y'])
    output_dht(f, 2, 0, compressed['huffman_tup_dc_c'])
    output_dht(f, 3, 1, compressed['huffman_tup_ac_c'])
    # 输出扫描行开始.
    output_sos(f, 0, 1, 2, 3, 2, 3)
    # 输出图像信息.
    output_mcus(f,
        compressed['dc_coded_y'],
        compressed['ac_coded_y'],
        compressed['dc_coded_cb'],
        compressed['ac_coded_cb'],
        compressed['dc_coded_cr'],
        compressed['ac_coded_cr']
    )
    # 输出文件尾巴
    output_eoi(f)
    f.flush(output_0xff_callback)

def main():
    img = cv2.imread("./img.png")
    img =cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    qua= 0.5
    jpeg(img, "img_compressed_{}.jpg".format(qua), qua)
    print("done")


if __name__ == '__main__':
    main()