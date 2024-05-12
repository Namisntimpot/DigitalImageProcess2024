import numpy as np
import re

# 使用numpy的random.randint函数生成随机整数
random_integers = np.random.randint(0, 256, size=64)  # 256是上界，不包含256本身，所以范围是0-255

print("Random integers from 0 to 255:", random_integers)

s = str(random_integers)
s = re.sub(r'\s+', ',', s)

print("\n", s)