import numpy as np
import cv2
from scipy.fft import dct, idct

def dct2d(img):
    dct_row = dct(img, axis=0)
    ret = dct(dct_row, axis=1)
    return ret

fname = "../image/dance_crop_gray"
img = cv2.imread(fname + ".jpg")[:,:,0]  # 灰度图，只取一个通道即可

dct_ret = dct2d(img)

amp = np.log(1 + np.abs(dct_ret))
amp = (amp - amp.min()) / (amp.max() - amp.min()) * 255

cv2.imwrite(fname + "_dct_py.jpg", amp)