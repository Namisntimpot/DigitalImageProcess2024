import os
import numpy as np

def bit_str_add1(s):
    ''' 不考虑进位，即全1的情况'''
    ret = ''
    i = -1
    while i >= -len(s):
        if s[i] == '0':
            ret = '1' + ret
            break
        else:
            ret = '0' + ret
            i-=1

    ret = s[0:i] + ret
    return ret

def bit_reverse_highest(s):
    return '0' + s[1:] if s[0] == '1' else '1' + s[1:]

def bit_reverse_all(s):
    ret = ''
    for c in s:
        ret += '1' if c == '0' else '0'
    return ret

def int_to_bit(num):
    if num == 0:
        return ''
    a = int(abs(num))
    wei = int(np.floor(np.log2(a)) + 1)
    ret = ''
    while wei > 0:
        ret = '1' + ret if a % 2 == 1 else '0' + ret
        a //= 2
        wei -= 1
    if num < 0:
        ret = bit_reverse_all(ret)
    return ret


class VLITable:
    def lookup(num):
        assert num < 2048
        if num==0:
            return 0, ''
        return np.floor(np.log2(abs(num))) + 1, int_to_bit(num)


class QuanTable:
    '''
    量化表，来自Photoshop - (Save For Web 080) \\
    (http://www.impulseadventure.com/photo/jpeg-quantization.html)
    '''
    Lum = np.array([[2, 2, 2, 2, 3, 4, 5, 6],
                      [2, 2, 2, 2, 3, 4, 5, 6],
                      [2, 2, 2, 2, 4, 5, 7, 9],
                      [2, 2, 2, 4, 5, 7, 9, 12],
                      [3, 3, 4, 5, 8, 10, 12, 12],
                      [4, 4, 5, 7, 10, 12, 12, 12],
                      [5, 5, 7, 9, 12, 12, 12, 12],
                      [6, 6, 9, 12, 12, 12, 12, 12]])
    Chrom = np.array([[3, 3, 5, 9, 13, 15, 15, 15],
                      [3, 4, 6, 11, 14, 12, 12, 12],
                      [5, 6, 9, 14, 12, 12, 12, 12],
                      [9, 11, 14, 12, 12, 12, 12, 12],
                      [13, 14, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12]])
    

def zigzag_scan(img:np.ndarray) -> np.ndarray:
    '''img需要是单通道的，shape=(h,w)'''
    assert len(img.shape) == 2
    h, w = img.shape
    x, y = 0, 0
    ret = []
    direction = 0  # direction: 0-向左或向下，1-向斜下方, 2-向斜上方.
    while x != w-1 or y != h-1:
        ret.append(img[y, x])
        if direction == 0:
            if y == 0:
                if x + 1 != w:  # 第一行非末尾，向左
                    x += 1
                else:   # 第一行末尾，向下
                    y += 1
                direction = 1
            elif x == 0:        # 第一列非左上角
                if y + 1 != h : # 第一列非左下角, 向下
                    y += 1
                else:           # 第一列左下角，向左
                    x += 1
                direction = 2
            elif y == h-1:      # 最后一行非左下角
                x += 1
                direction = 2
            elif x == w - 1:    # 最后一列非右上角
                y += 1
                direction = 1
        elif direction == 1:
            y += 1
            x -= 1
            if x == 0 or y == h-1:
                direction = 0
        elif direction == 2:
            y -= 1
            x += 1
            if y == 0 or x == w-1:
                direction = 0
    ret.append(img[h-1, w-1])
    return np.array(ret)


if __name__ == '__main__':
    a = np.random.random((8,8))
    print("")
    print(a)
    print(zigzag_scan(a))