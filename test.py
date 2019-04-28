# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:07:32 2019

@author: chenjie
"""


import os.path as osp
import sys
import numpy as np
import pandas as pd # 用来导入和导出csv数据特别方便
import random
import time

# 打日志
import logging
import logging.handlers

from load_data import load_data 
from user_item_mapping import user_item_mapping



LOG_FILE = "train_log.log"
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes = 1024*1024*1024*1024, backupCount = 5)
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'

formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)


logger = logging.getLogger('train_log')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.info("first info message")
logger.debug("first debug message")


"""
损失函数
Root mean suqure deviation(偏差)
y_e: estimation of y
y_r: real value of y
"""
def RMSD_loss(y_e, y_r):
    if len(y_e)!=len(y_r):
        print("长度不一致，出错啦!y_e长度:%d,y_r长度:%d"%(len(y_e), len(y_r)))
        sys.exit(1)
    T = len(y_e)
    sum = 0
    for idx in range(T):
        y1 = y_e[idx]
        y2 = y_r[idx]
        sum = np.add(sum, np.power(y1-y2,2))
    sum = np.divide(sum, T)
    return np.sqrt(sum)

"""
data_set为train_set或者validation_set
计算出P,Q矩阵在训练集或者是测试集上的均方损失
"""
def loss(data_set, p,q):
    data_set = np.array(data_set)
    y_r = data_set[:,2] # real rating
    y_e = []
    for row in data_set:
        u = row[0] # user_id
        i = row[1] # item_id
        y_e.append(Predict(u,i,p,q))
    return RMSD_loss(y_e,y_r)


def infer_on_test():
    pass

"""
train: 训练数据集，含有很多[u,i,r_ui]的打分数据
        csv文件里面如:

val:   验证集            
F:     期望的降维维度
totalSteps:     总共的迭代次数
alpha: 初始的步长
_lambda:     正则化项
saveModelIter: 每训练多少步保存一次矩阵P，Q
"""
def LearningLFM(train, val, F, totalSteps, alpha, _lambda, saveModelIter):
    ori_alpha = alpha
    ori_lambda = _lambda
    p,q = InitLFM(train,F)
    N = len(train)
    print("数据总量有%d条"%(N))
    """一次的epoch刚好迭代了N次，所有的epoches跑完"""
    """和totalSteps比较接近，但是也不一定，因为有一个整除在这里"""
    stepCount = 0 # 记录总共迭代的次数
    epoches = int(totalSteps+N-1 / N)
    for epoch in range(epoches):
        directions = [i for i in range(N)]
        random.shuffle(directions)
        """进行随机梯度下降"""
        for idx in directions: #
            stepCount += 1
            if stepCount % 1000 == 0:
                msg = "[Prompt]Epoch:%d, Current step: %d, Loss(train): %f, Loss(val): %f, alpha:%f"%(epoch, stepCount, loss(train,p,q), loss(val,p,q), alpha)
                print(msg)
                logger.info(msg)

            if stepCount%saveModelIter==0:
                msg = "_F_%d_ori_alpha_%f_ori_lambda_%f"%(F,ori_alpha,ori_lambda)
                save_model(p,"P",stepCount, msg)
                save_model(q,"Q", stepCount, msg)
                
            if stepCount>totalSteps:
                msg = "【训练结束】"
                print(msg)
                logger.info(msg)
                return p,q
                        
            [u,i,rui] = train[idx]
            pui = Predict(u,i,p,q)
            eui = rui - pui 
            for k in range(F): # 0,1,2, ..., F-1
                p[u][k] += alpha * (q[i][k]*eui - _lambda * p[u][k])
                q[i][k] += alpha * (p[u][k]*eui - _lambda * q[i][k])
        alpha = alpha * 0.9
    return p,q

"""
返回p,q矩阵
p,q的列数都是一样的，都是F列
p有m行，m为user的总人数
q有n行，n为item的总个数
"""
def InitLFM(train, F):
    p = dict()
    q = dict()
    N = len(train)
    for idx in range(N):
        [u,i,rui] = train[idx]
        if u not in p:
            p[u] = [random.random()/np.sqrt(F) for x in range(F)]
        if i not in q:
            q[i] = [random.random()/np.sqrt(F) for x in range(F)]

    return p,q

"""
在给定模型p和q时，给出用户u在物品i上的输出，是实数，不是整数
"""
def Predict(u,i,p,q):
    return sum([p[u][f]*q[i][f] for f in range(len(p[u]))])
    



"""
Label: "P" or "Q"
iter: iteration steps, 1000,10000, etc
"""
def save_model(M,Label,iter,msg):
    M = pd.DataFrame(M)
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    filename = "data\\matrix\\%s_iter%d_%s_%s.csv"%(Label,iter,msg,time_str)
    M.to_csv(filename, header = False, index=False)
    msg = "\n[Saved Matrix %s to file: %s]\n"%(Label, filename)
    print(msg)
    logger.info(msg)
"""
在测试集上给测试集打分
输出的是整数
"""
def rate_on_test(p,q):
    test = np.array(pd.read_csv("data\\test_set.csv", header = None))
    test = list(test)
    user_id_list = np.array(pd.read_csv("data\\user_id_mapping.csv", header = None))
    item_id_list = np.array(pd.read_csv("data\\item_id_mapping.csv", header = None))
    N = len(test)
    for idx in range(N):
        u = test[idx][0]
        i = test[idx][1]
        rui = Predict(u,i,p,q)
        test[idx].append(rui)
        
    for idx in range(N):
        u = test[idx][0]
        i = test[idx][1]
        test[idx][0] = user_id_list[u]
        test[idx][1] = item_id_list[i]
    test =pd.DataFrame(test)
    
    # 预测值的输出评分
    test.to_csv("data\\prediction_result.csv", header = False, index=False)


def main():
    # test filename
    test_fn = "test_set.csv"
    # training filename
    train_fn = "training_set.csv"
    # validation filename
    val_fn = "validation_set.csv"
    
    train = np.array(pd.read_csv(osp.join('data', train_fn), header = None))
    val = np.array(pd.read_csv(osp.join('data',val_fn), header = None))
    F = 20
    totalSteps = 1000000
    saveModelIter = 5000
    alpha = 0.03
    _lambda = 0.5
    msg = "[start_again]_F_%d_alpha_%f_lambda_%f"%(F,alpha,_lambda)
    logger.info(msg)
    p,q = LearningLFM(train,val, F,totalSteps,alpha,_lambda,saveModelIter)
    p = pd.DataFrame(p)
    p.to_csv("P_iter%d.csv"%(totalSteps), header = False,index=False)
    q = pd.DataFrame(q)
    q.to_csv("Q_iter%d.csv"%(totalSteps), header = False,index=False)
    
if __name__ == "__main__":
    """数据预处理，只需要执行一次,把结果保存在文件里面就可以了"""
    # user_item_mapping() 
    main()
    

    