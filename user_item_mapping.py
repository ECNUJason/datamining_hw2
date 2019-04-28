import numpy as np
import pandas as pd # 用来导入和导出csv数据特别方便
from load_data import load_data

HAS_EXECUTED = True
"""
只需要执行一次，就把所有的数据都处理好了
接下来只需要处理这些文件里面的数据就好了：
data/
    test_set.csv
    training_set.csv
    validation_set.csv
    user_id_mapping.csv
    item_id_mapping.csv

user_id到0，1，2，....， m-1的转换
一共有m个user，但是这里的userid有些乱，所以打算做一个映射
并且保存到文件里面去
发现测试集里面，用户一共有323个

还做了item_id到0，1，2，....,n-1的转换
其中n是所有的item的总数

然后把training_set里面的user_id和item_id全都换成了0,1,2,...，并且删除了评论的内容

对应的还有validation_set和test_set

"""
def user_item_mapping():
    if HAS_EXECUTED:
        print("[user_item_mapping.py]已经执行过啦,如果还想再执行一次，可以手工修改HAS_EXECUTED参数")
        return None
    train_fn = "training_set.ss"
    # 为list
    data = load_data(train_fn)[1] # 获取数据
    user_id_list = []
    user_id_set = set()
    for row in data:
        user_id = row[0]
        if not user_id in user_id_set:
            user_id_list.append(user_id)
            user_id_set.add(user_id)
    user_id_set = None
    
    item_id_list = []
    item_id_set = set()
    
    
    for row in data:
        item_id = row[1]
        if not item_id in item_id_set:
            item_id_list.append(item_id)
            item_id_set.add(item_id)
    item_id_set = None
    
    for i in range(len(data)):
        row = data[i]
        user_id = row[0]
        item_id = row[1]
        data[i][0] = user_id_list.index(user_id)
        data[i][1] = item_id_list.index(item_id)
    data = np.array(data)
    data = data[:,0:-1]
    data = pd.DataFrame(data)
    data.to_csv("data\\training_set.csv",header = False, index=False)
    
    
    
    data = None
    
    val_fn = "validation_set.ss"
    data = load_data(val_fn)[1]
    for i in range(len(data)):
        row = data[i]
        user_id = row[0]
        item_id = row[1]
        data[i][0] = user_id_list.index(user_id)
        data[i][1] = item_id_list.index(item_id)
    data = np.array(data)
    data = data[:,0:-1]
    data = pd.DataFrame(data)
    data.to_csv("data\\validation_set.csv",header = False, index=False)
    data = None
    
    test_fn = "test_set.ss"
    data = load_data(test_fn)[1]
    for i in range(len(data)):
        row = data[i]
        user_id = row[0]
        item_id = row[1]
        data[i][0] = user_id_list.index(user_id)
        data[i][1] = item_id_list.index(item_id)
    data = np.array(data)
    data = data[:,0:-1]
    data = pd.DataFrame(data)
    data.to_csv("data\\test_set.csv",header = False, index=False)
    data = None
    """然后把这个对应关系也导出"""
    user_id_list = pd.DataFrame(user_id_list)
    user_id_list.to_csv("data\\user_id_mapping.csv", header = False,index=False)
    
    item_id_list = pd.DataFrame(item_id_list)
    item_id_list.to_csv("data\\item_id_mapping.csv", header = False,index=False)
    
    user_id_list = None
    item_id_list = None
    
"""
按顺序排列，排在第一个的，那么就是0,然后是1号，2号，依次类推
"""
def TEST_user_item_mapping():
    A = np.array(pd.read_csv("data\\item_id_mapping.csv", header = None))
    print(A)