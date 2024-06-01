import os
import sys

class BitOutStream:
    def __init__(self, filepath) -> None:
        '''
        按byte输出. 大端法.
        '''
        self.f = open(filepath, 'wb')
        self.buf = ''  # 存储01字符串的buf...
    
    def __del__(self):
        self.f.close()

    def write(self, out, callback = None):
        '''
        out要么是0-1字符串，要么直接是bytes.  \\
        如果接收到了bytes，会把buf中不足8bits的在低位填充0，然后直接输出bytes.  \\
        如果收到了0-1字符串，只会放到buf里，不会直接输出，要flush了再输出！  \\
        如果callback不是None，会在输出每个byte的时候调用它.  \\
        callback: int->list[int]   \\
        它的用法是为了应对：如果数据中有0xFF，就在后面补0x00.
        '''
        if type(out) == bytes:
            self.flush(callback=callback)
            if callback is not None:
                to_print = []
                for a in out:
                    to_print += callback(a)
                self.f.write(bytes(to_print))
            else:
                self.f.write(out)
        elif type(out) == str:
            self.buf += out
        else:
            raise "out只支持bytes或者0-1字符串"
        

    def flush(self, callback = None):
        '''
        清空buf，不满8bits的在低位填0. \\
        如果callback不是None，会在输出每个byte的时候调用它.
        '''
        if len(self.buf) == 0:
            return
        i = 0
        bytes_num_arr = []
        while i + 8 <= len(self.buf):
            b = self.buf[i:i+8]
            if callback is None:
                bytes_num_arr.append(int(b, 2))
            else:
                bytes_num_arr += callback(int(b, 2))
            i += 8
        if i < len(self.buf):
            tail = self.buf[i:]
            n_pad = 8-len(tail)
            b = tail + '0'*n_pad
            if callback is None:
                bytes_num_arr.append(int(b,2))
            else:
                bytes_num_arr += callback(int(b, 2))
        self.f.write(bytes(bytes_num_arr))
        self.buf = ''


def output_0xff_callback(num):
    if num == 0xFF:
        return [0xFF, 0x00]
    else:
        return [num]