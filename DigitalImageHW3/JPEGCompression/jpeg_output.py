from bit_outstream import *
import numpy as np
from tqdm import tqdm
'''
参考：https://blog.csdn.net/yun_hen/article/details/78135122
jpeg格式组成部分：
SOI（文件头）+APP0（图像识别信息）+ DQT（定义量化表）+ SOF0（图像基本信息）+ DHT（定义Huffman表） + DRI（定义重新开始间隔）+ SOS（扫描行开始）+ EOI（文件尾）
'''

def output_soi(f:BitOutStream):
    '''
    输出jpeg文件头0xFF, 0xD8
    '''
    f.write(bytes([0xFF, 0xD8]))

def output_app0(
        f:BitOutStream,
        major_version = 1,
        minor_version = 1,
        ppi_unit = 1,   # 0-无单位，1-点数/英寸，2-点数/厘米.
        ppi_x = 96,     # 宽方向的像素密度, 2B, 小于65535
        ppi_y = 96,     # 高方向的像素密度, 2B, <=65535
    ):
    '''
    输出图像识别信息APP0节, 标识为0xFF, 0xE0  \\
    无缩略图
    '''
    # 段头
    f.write(bytes([0xFF, 0xE0]))
    # 段长度，JFIF, 主版本号，次版本号，像素单位，x像素密度，y像素密度，缩略图x像素数，y像素数.
    content = bytes([0x00, 0x10]) + bytes([0x4A,0x46,0x49,0x46,0x00]) + bytes([major_version,minor_version,ppi_unit]) + \
                ppi_x.to_bytes(2, 'big') + ppi_y.to_bytes(2, 'big') + bytes([0,0])
    f.write(content, callback=output_0xff_callback)

def output_dqt(
        f:BitOutStream,
        QTable:np.ndarray,
        QT_number,
        QT_8bit = True,
        endian_swap = True
    ):
    '''
    输出一个只有一个量化表的DQT段。如果是16bit，需要用endian_swap指定是否要切换端序，注意numpy是小端法，而jpeg是大端法.
    '''
    if QT_8bit:
        bytes_table = bytes(np.uint8(QTable.flatten()))
    else:
        bytes_table = bytes(np.uint16(QTable.flatten()).byteswap() if endian_swap else np.uint16(QTable.flatten()))
    seg_length = 3 + 64 * (1 if QT_8bit else 2)
    f.write(bytes([0xFF, 0xDB]))
    content = seg_length.to_bytes(2,'big') + bytes([0 if QT_8bit else 1, QT_number]) + bytes_table
    f.write(content, output_0xff_callback)

def output_sof0(
        f:BitOutStream,
        height:int, weight:int,
        QTable_number_Y, QTable_numer_C,  # 量化表号.
        vertical_sample_coef_Y = 2,
        horizontal_sample_coef_Y = 2,
        vertical_sample_coef_Cb = 2,
        horizontal_sample_coef_Cb = 2,
        vertical_sample_coef_Cr = 2,
        horizontal_sample_coef_Cr = 2,
        precision = 8
    ):
    '''
    输出图像基本信息段, 只支持YCrCb., 只支持8bit.  \\
    由于没考虑422,420这些东西，此前考虑的MCU是一个8*8，Y,Cb,Cr分块各一个，应该是444格式，不支持其它.否则可能有问题.
    '''
    assert precision == 8, "目前只支持8bit精度"

    f.write(bytes([0xFF, 0xC0]))  # 段表示符
    seg_length = bytes([0, 8+3*3])  # YCbCr三个组件 段长度是两字节!
    seg_sample_precision = precision.to_bytes(1,'big')
    seg_height = height.to_bytes(2,'big')
    seg_width = height.to_bytes(2, 'big')
    seg_num_component = bytes([3])
    seg_Y_component = bytes([1, (horizontal_sample_coef_Y<<4)+vertical_sample_coef_Y, QTable_number_Y])
    seg_Cb_component= bytes([2, (horizontal_sample_coef_Cb<<4)+vertical_sample_coef_Cb, QTable_numer_C])
    seg_Cr_component= bytes([3, (horizontal_sample_coef_Cr<<4)+vertical_sample_coef_Cr, QTable_numer_C])
    
    content = seg_length + seg_sample_precision + seg_height + seg_width + seg_num_component + seg_Y_component + seg_Cb_component + seg_Cr_component
    f.write(content, output_0xff_callback)


def output_dht(
        f:BitOutStream,
        HT_number,
        HT_type,  # 0-DC表, 1-AC表
        HT_tuple
    ):
    '''
    输出一个含有一个哈夫曼表的DHT段  \\
    HT_tupe: list of (val, code)，且code是按照jpeg要求排好序的.  \\
    因为现在不会把huffman码长限制在16以内，所以**姑且把bit_table加到24位**...
    '''
    f.write(bytes([0xFF, 0xC4]))

    ext_bit_table_length = 24  # 最大码长姑且加到24...

    seg_ht_info = bytes([(HT_type << 4) + HT_number])
    # tuple_list：是(val, code)
    seg_ht_bit_table = [0 for i in range(ext_bit_table_length)]
    seg_ht_val_table = []
    for val, code in HT_tuple:
        seg_ht_bit_table[len(code) - 1] += 1
        seg_ht_val_table.append(np.uint8(val))
    seg_length = 19 + len(seg_ht_val_table) + ext_bit_table_length - 16
    seg_length = seg_length.to_bytes(2, 'big')
    seg_ht_bit_table = bytes(seg_ht_bit_table)
    seg_ht_val_table = bytes(seg_ht_val_table)
    
    content = seg_length + seg_ht_info + seg_ht_bit_table + seg_ht_val_table
    f.write(content, output_0xff_callback)

def output_sos(
        f:BitOutStream,
        huffman_dc_num_Y, huffman_ac_num_Y,
        huffman_dc_num_Cb,huffman_ac_num_Cb,
        huffman_dc_num_Cr,huffman_ac_num_Cr
    ):
    '''
    输出一个含有Y,Cb,Cr信息的SOS段.
    '''
    f.write(bytes([0xFF, 0xDA]))
    seg_length = (6 + 2*3).to_bytes(2, 'big')
    seg_num_components = bytes([3])
    seg_component_y = bytes([1, (huffman_dc_num_Y<<4) + huffman_ac_num_Y])
    seg_component_cb= bytes([2, (huffman_dc_num_Cb<<4) + huffman_ac_num_Cb])
    seg_component_cr= bytes([3, (huffman_dc_num_Cr<<4) + huffman_ac_num_Cr])
    seg_padding = bytes([0,0,0]) # 最后空3个全0的不知什么用的字节.
    
    content = seg_length + seg_num_components + seg_component_y + seg_component_cb + seg_component_cr + seg_padding
    f.write(content, output_0xff_callback)

def output_mcus(
        f:BitOutStream,
        y_dc_huffman,
        y_ac_huffman,
        cb_dc_huffman,
        cb_ac_huffman,
        cr_dc_huffman,
        cr_ac_huffman,
        n_elem_per_mcu = 64
    ):
    '''
    输出所有8*8 mcu 的信息，Y->Cb->Cr，每个分量第一个是dc，后面全是ac.  \\
    注意，ac已经被flatten 了. 
    '''
    assert n_elem_per_mcu == 64
    
    n_mcus = len(y_dc_huffman)
    for dc_huf, ac_huf in zip([y_dc_huffman,cb_dc_huffman,cr_dc_huffman],[y_ac_huffman,cb_ac_huffman,cr_ac_huffman]):
        for i in tqdm(range(n_mcus)):
            dc = dc_huf[i]
            ac = ac_huf[i * (n_elem_per_mcu-1) : (i+1)*(n_elem_per_mcu-1)]  # 每个mcu少了左上角那个!
            f.write(dc[0]+dc[1], output_0xff_callback)
            for a,b in ac:
                f.write(a+b, output_0xff_callback)
    f.flush(output_0xff_callback)

def output_eoi(f:BitOutStream):
    f.write(bytes([0xFF, 0xD9]))