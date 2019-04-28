# -*- coding: utf-8 -*-
import os
import os.path as osp
import numpy as np

BASE_FP = osp.join(osp.dirname(os.path.abspath(__file__)),"data")

"""
返回的是2个python内置的数据类型list
专门用来导入ss文件。
因为ss文件开头有header，而csv文件都没有
所以干脆分开成不同的方法来导入不同的数据集
而且，csv的文件，使用pandas来导入相对来说也是很简单的了
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

