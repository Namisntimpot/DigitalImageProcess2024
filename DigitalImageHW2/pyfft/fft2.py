import numpy as np
import cv2

fname = "../image/Goldhill"
img = cv2.imread(fname + ".jpg")[:,:,0]  # 灰度图，只取一个通道即可

fft_ret = np.fft.fft2(img)
fft_shifted = np.fft.fftshift(fft_ret)

amp = np.log(np.abs(fft_shifted) + 1)
amp = (amp - amp.min()) / (amp.max() - amp.min()) * 255

cv2.imwrite(fname + "_fft_py.jpg", amp)