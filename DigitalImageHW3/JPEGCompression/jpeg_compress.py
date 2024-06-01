import os
import numpy as np
import scipy
import cv2

from huffman import *
from utils import *

def jpeg_compress(img:np.ndarray, quality_coef = 1):
    '''
    输入YCbCr格式的图片, 暂时不考虑长宽不是8的倍数
    '''
    height, width, _ = img.shape
    assert height % 8 == 0 and width % 8 == 0
    
    Y_component = img[:,:,0]
    Cb_component = img[:,:,1]
    Cr_component = img[:,:,1]

    n_blks_h, n_blks_w = height // 8, width // 8
    n_blks = n_blks_w * n_blks_h

    DC_Y_ORI = []   # 将会是(n_blks)，存放每个量化后的Y块的直流分量的原始形式
    AC_Y_ORI = []   # 将会是(n_blks, 63), 存放每个量化后的Y块的交流分量的原始数值形式. 但是后续会.flatten，直接展平.
    DC_CB_ORI, AC_CB_ORI = [], []
    DC_CR_ORI, AC_CR_ORI = [], []

    l_mcu = 8

    # 对每个块进行离散傅里叶变换，并且直接量化.
    for comp, dclist, aclist, ty in zip(
        [Y_component, Cb_component, Cr_component],
        [DC_Y_ORI, DC_CB_ORI, DC_CR_ORI],
        [AC_Y_ORI, AC_CB_ORI, AC_CR_ORI],
        ['Y', 'C', 'C']
        ):
        for y in range(n_blks_h):
            for x in range(n_blks_w):
                blk = comp[y*l_mcu:(y+1)*l_mcu, x*l_mcu:(x+1)*l_mcu].astype(np.float32)
                dct_ret = cv2.dct(blk)
                # 进行量化.
                if ty == 'Y':
                    quan_ret = np.round(dct_ret / np.uint(QuanTable.Lum / quality_coef))
                else:
                    quan_ret = np.round(dct_ret / np.uint(QuanTable.Chrom / quality_coef))
                # zigzag扫描，成为一维数组
                zigzag_ret = zigzag_scan(quan_ret)
                # 保存直流和交流分量.
                dclist.append(zigzag_ret[0])
                aclist.append(zigzag_ret[1:])
    # 变为np数组
    DC_Y_ORI, DC_CB_ORI, DC_CR_ORI, AC_Y_ORI, AC_CB_ORI, AC_CR_ORI = np.array(DC_Y_ORI),\
            np.array(DC_CB_ORI),np.array(DC_CR_ORI),np.array(AC_Y_ORI).flatten(),np.array(AC_CB_ORI).flatten(),np.array(AC_CR_ORI).flatten()
    
    # 处理直流分量
    # 对DC分量做差分
    DC_Y_ORI[1:] = DC_Y_ORI[1:] - DC_Y_ORI[:-1]
    DC_CB_ORI[1:] = DC_CB_ORI[1:] - DC_CB_ORI[:-1]
    DC_CR_ORI[1:] = DC_CR_ORI[1:] - DC_CR_ORI[:-1]
    # 对DC查表编码并转化为二元组 (长度，编码).
    DC_Y_MID, DC_CB_MID, DC_CR_MID = [], [], []
    for dc, mid in zip([DC_Y_ORI, DC_CB_ORI, DC_CR_ORI], [DC_Y_MID, DC_CB_MID, DC_CR_MID]):
        for i in range(dc.shape[0]):
            mid.append(VLITable.lookup(dc[i]))
    # 有了中间表达后，把每个元组的第一项拿出来，将要用于哈夫曼编码
    HUF_DC_Y = get_array_for_huffman(DC_Y_MID)
    HUF_DC_C = get_array_for_huffman(DC_CB_MID + DC_CR_MID)
    # 进行哈夫曼编码
    HUF_DICT_DC_Y, HUF_TUP_DC_Y = ComputeHuffmanCodeFromScratch(HUF_DC_Y)
    HUF_DICT_DC_C, HUF_TUP_DC_C = ComputeHuffmanCodeFromScratch(HUF_DC_C)
    # 进行编码
    DC_Y_HUFFMAN = get_huffman_code(DC_Y_MID, HUF_DICT_DC_Y)
    DC_CB_HUFFMAN = get_huffman_code(DC_CB_MID, HUF_DICT_DC_C)
    DC_CR_HUFFMAN = get_huffman_code(DC_CR_MID, HUF_DICT_DC_C)

    # 处理交流分量
    # 计算行程编码RLC, (几个0，这个数是几), 转化为二元组(前面有几个0，这个数是多少)
    RLC_AC_Y = compute_rlc(AC_Y_ORI.tolist())
    RLC_AC_CB = compute_rlc(AC_CB_ORI.tolist())
    RLC_AC_CR = compute_rlc(AC_CR_ORI.tolist())
    # 把形成编码转化为中间形式, 几个0和编码长度**已经拼接**
    AC_Y_MID = get_ac_mid_from_rlc_array(RLC_AC_Y)
    AC_CB_MID = get_ac_mid_from_rlc_array(RLC_AC_CB)
    AC_CR_MID = get_ac_mid_from_rlc_array(RLC_AC_CR)
    # 进行霍夫曼编码
    HUF_DICT_AC_Y, HUF_TUP_AC_Y = ComputeHuffmanCodeFromScratch(get_array_for_huffman(AC_Y_MID))
    HUF_DICT_AC_C, HUF_TUP_AC_C = ComputeHuffmanCodeFromScratch(get_array_for_huffman(AC_CB_MID + AC_CR_MID))
    # 执行编码
    AC_Y_HUFFMAN = get_huffman_code(AC_Y_MID, HUF_DICT_AC_Y)
    AC_CB_HUFFMAN = get_huffman_code(AC_CB_MID, HUF_DICT_AC_C)
    AC_CR_HUFFMAN = get_huffman_code(AC_CR_MID, HUF_DICT_AC_C)

    ret = {
        'huffman_dict_dc_y':    HUF_DICT_DC_Y,
        'huffman_dict_dc_c':    HUF_DICT_DC_C,
        'huffman_dict_ac_y':    HUF_DICT_AC_Y,
        'huffman_dict_ac_c':    HUF_DICT_AC_C,
        'huffman_tup_dc_y':     HUF_TUP_DC_Y,
        'huffman_tup_dc_c':     HUF_TUP_DC_C,
        'huffman_tup_ac_y':     HUF_TUP_AC_Y,
        'huffman_tup_ac_c':     HUF_TUP_AC_C,
        'dc_coded_y':           DC_Y_HUFFMAN,
        'dc_coded_cb':          DC_CB_HUFFMAN,
        'dc_coded_cr':          DC_CR_HUFFMAN,
        'ac_coded_y':           AC_Y_HUFFMAN,
        'ac_coded_cb':          AC_CB_HUFFMAN,
        'ac_coded_cr':          AC_CR_HUFFMAN
    }
    return ret


def get_array_for_huffman(arr):
    '''
    从DC_MID中提取要用于哈弗曼编码的项(每个元组中的第一个)
    '''
    return [x[0] for x in arr]

def get_ac_mid_from_rlc_array(arr):
    '''
    把每个元素元组的第二个数字查VIL表, 变为((有几个0，编码多长)(已拼接)，编码是几(01字符串))
    '''
    mid = []
    for n_zeros, num in arr:
        l,c = VLITable.lookup(num=num)
        mid.append(((np.uint8(n_zeros)<<4) + np.uint8(l), c))
    return mid

def compute_rlc(arr):
    '''
    计算AC分量的RLC编码，返回的每项是(前有几个0，这个数是几). \\
    如果有连续16个0，需要用(15, 0)；如果某个数字后面全是0，用EOB，即(0,0)表示.
    '''
    # 先找到最后一个非0的数. 必须有非0的数.
    last_nonzero = -1
    while arr[last_nonzero] == 0 and last_nonzero >= -len(arr):
        last_nonzero -= 1
    assert last_nonzero >= -len(arr)
    last_nonzero += len(arr)

    rlc = []
    n_zeros = 0
    for i in range(len(arr)):
        if i > last_nonzero:
            rlc.append((0,0))
            break
        num = arr[i]
        if num == 0:
            if n_zeros == 15:
                # 出现了连续16个0.
                rlc.append((15,0))
                n_zeros = 0
            else:
                n_zeros += 1
        else:
            rlc.append((n_zeros, num))
            n_zeros = 0
    return rlc

def get_huffman_code(arr, d):
    ret = []
    for num, vli in arr:
        ret.append((d[num], vli))
    return ret