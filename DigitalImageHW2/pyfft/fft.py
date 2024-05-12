import numpy as np
from scipy.fft import dct, idct, dctn, idctn

a = np.array([197,111,67,137,89,38,19,252,95,213,28,57,187,200,98,166,153,123,221,4,102,186,87,16,45,244,132,79,100,213,210,189,28,216,181,210,71,3,127,237,86,244,254,178,10,209,215,206,134,212,206,212,113,7,48,69,252,27,220,131,18,80,38,44])
a = a.reshape((8,8))
#a = np.array([1,2,3,4,5,6,7,8])
#arr = np.array([1,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0])
arr = np.concatenate((a, np.zeros(a.shape)))
#f = np.fft.fft(arr)
#arr = arr.reshape((8,8))

#f = np.fft.fft2(arr)
# c = np.cos(np.pi * np.arange(len(a)) / len(a) / 2)
# s = np.sin(np.pi * np.arange(len(a)) / len(a) / 2)
# f = np.fft.fft(arr)#.real[:len(a)] * 2 * c
# f = f.real[:len(a)] * 2 * c + f.imag[:len(a)] * 2 * s
dct2d_tmp = dct(a, axis=0)
d = dct(dct2d_tmp, axis=1)
#i = dct(d, type=3) / 2 / a.shape[0]
#print(f)
print(d)
#print(i)