# -*- coding: utf-8 -*-
# @Time    : 2019/11/9 20:53
# @Author  : tys
# @Email   : yangsongtang@gmail.com
# @File    : test_3.py
# @Software: PyCharm

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr.mean())
print(arr.mean(0))
print(arr.mean(1))