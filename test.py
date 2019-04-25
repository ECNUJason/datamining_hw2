# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:07:32 2019

@author: chenjie
"""

import os
import os.path as osp
import sys
import numpy as np
import pandas as pd


BASE_FP = osp.join(osp.dirname(os.path.abspath(__file__)),"data")

def load_data(filename=None):
    if filename == None:
        filename = "validation_set.ss"
    data_fp = osp.join(BASE_FP,filename)
    print(data_fp)
    data = np.genfromtxt(data_fp,
                         delimiter='\t',
                         dtype=str,
                         encoding="UTF-8")
    return data[0], data[1:] # header and the rest

def train():
    pass

def val():
    pass

def infer_on_test():
    pass

def main():
    # test filename
    test_fn = "test_set.ss"
    # training filename
    train_fn = "training_set.ss"
    # validation filename
    val_fn = "validation_set.ss"
    
    
    
if __name__ == "__main__":
    main()
    
    