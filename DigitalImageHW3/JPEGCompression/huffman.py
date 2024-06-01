import numpy as np
from queue import PriorityQueue, Queue
from utils import *

def ComputeHuffmanCodeFromScratch(arr:list) -> dict:
    '''
    输入：一个数组，要对数组中的元素编码 \\
    输出：一个字典 dict: elem_in_arr -> str，str是01串；一个列表，每个元素是(val, coding)元组，按元组升序排列. \\
    如何确保最后每个码长不超过16? 只使用标准码表？
    '''
    class __node:
        def __init__(self, val, w, left_node, right_node):
            self.val = val
            self.w = w
            self.left_node = left_node
            self.right_node= right_node

        def is_leaf(self):
            return self.left_node is None and self.right_node is None
    
    freq = {}
    for elem in arr:
        try:
            freq[elem] += 1
        except:
            freq[elem] = 1
    forest = []
    for k, v in freq.items():

        forest.append(__node(k, v, None, None)) 

    if len(forest) == 1:
        coding_dict = {forest[0].val : '0'}
        coding_tuple = [(forest[0].val, '0')]
        return coding_dict, coding_tuple
    
    forest.sort(key=lambda x: x.w)
    length = len(forest)
    while length != 1:
        l = forest[0]
        r = forest[1]
        new_node = __node(-1, l.w + r.w, l, r)
        i = 2
        placed = False
        while i < length:
            if new_node.w < forest[i].w:
                forest[i-2] = new_node
                placed = True
                break
            else:
                forest[i-2] = forest[i]
                i += 1
        while i < length:
            forest[i-1] = forest[i]
            i += 1
        if not placed:
            forest[length-2] = new_node
        length -= 1
        
    coding_dict = {}
    coding_tuple = []
    traverse_queue = Queue()
    traverse_queue.put((forest[0], ''))
    while not traverse_queue.empty():
        node, code = traverse_queue.get()
        if node.is_leaf():
            coding_dict[node.val] = code
            coding_tuple.append((node.val, code))
        else:
            if node.left_node is not None:
                traverse_queue.put((node.left_node, code + '0'))
            if node.right_node is not None:
                traverse_queue.put((node.right_node, code + '1'))
    
    coding_tuple.sort(key=lambda x: len(x[1]))

    # 按照jpeg的哈夫曼编码的标准，重新调整哈夫曼编码.
    cur_len = len(coding_tuple[0][1])
    coding_dict[coding_tuple[0][0]] = '0' * cur_len
    coding_tuple[0] = (coding_tuple[0][0], '0' * cur_len)
    for i in range(1, len(coding_tuple)):
        k, c = coding_tuple[i]
        if len(c) > cur_len:
            cur_len = len(c)
            c = bit_str_add1(coding_tuple[i-1][1]) + '0'
        else:
            c = bit_str_add1(coding_tuple[i-1][1])
        coding_tuple[i] = (k, c)
        coding_dict[k] = c

    print(coding_tuple)

    return coding_dict, coding_tuple


if __name__ == '__main__':
    arr = [1,2,2,3,3,3,4,4,4,5,5,5,6,6,6,6,7,7,7,7,7,7,8,8,9,9,9,9]
    d, t = ComputeHuffmanCodeFromScratch(arr)
    print(d)
    print(t)