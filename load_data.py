# -*- coding: utf-8 -*-
import os
import os.path as osp
import numpy as np

BASE_FP = osp.join(osp.dirname(os.path.abspath(__file__)),"data")

"""
返回的是2个python内置的数据类型list
"""
def load_data(filename=None):
    if filename == None:
        filename = "validation_set.ss"
    data_fp = osp.join(BASE_FP,filename)
    print(data_fp)
    data = np.genfromtxt(data_fp,
                         delimiter='\t',
                         dtype=str,
                         encoding="UTF-8") # 如果没有UTF-8会默认用gbk格式打开，奇怪
    return data[0], data[1:] # header and the rest

