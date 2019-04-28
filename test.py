# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:07:32 2019

@author: chenjie
"""

import os
import os.path as osp
import sys
import numpy as np
import pandas as pd # 用来导入和导出csv数据特别方便


from load_data import load_data 
from user_item_mapping import user_item_mapping

"""
损失函数
Root mean suqure deviation(偏差)
y_e: estimation of y
y_r: real value of y
"""
def RMSD_loss(y_e, y_r):
    if len(y_e)!=len(y_r):
        print("长度不一致，出错啦!")
        sys.exie(1)
    T = len(y_e)
    sum = 0
    for idx in range(T):
        y1 = y_e[idx]
        y2 = y_r[idx]
        sum = np.add(sum, np.power(y1-y2,2))
    sum = np.divide(sum, T)
    return np.sqrt(sum)


def train():
    pass

def val():
    pass

def infer_on_test():
    pass

"""
train: 训练数据集，含有很多[u,i,r_ui]的打分数据
        csv文件里面如:
            
F:     期望的降维维度
N:     总共的迭代次数
alpha: 初始的步长
l:     正则化项
"""
def learningLFM(train, F, N, alpha, l):
    pass





    

def main():
    # test filename
    test_fn = "test_set.ss"
    # training filename
    train_fn = "training_set.ss"
    # validation filename
    val_fn = "validation_set.ss"
    
    print(load_data(val_fn)[0])
    
    
if __name__ == "__main__":
    """数据预处理，只需要执行一次,把结果保存在文件里面就可以了"""
    # user_item_mapping() 
    
    