# -*- coding: utf-8 -*-
'''
用于生成各个data集的attri_vector
将1,0,4.0,"[1, 2, 3, 4, 8]"改成1,0,4.0,[ 0  2  4  6  8 10 12 15 16 18 20 22 24 26 28 30 32 34]
从train/test/negtive_data.csv 中产生最终喂进网络的final_train/test/negtive_data.csv

'''

import numpy as np
import pandas as pd

#注意，真正喂进网络的数据并不需要rating，rating只用来划分用户
train_data = np.array(pd.read_csv(r'intermediate_data/train_data.csv', usecols=['userId', 'itemID', 'genre']))
neg_data = np.array(pd.read_csv(r'intermediate_data/neg_data.csv', usecols=['userId', 'itemID', 'genre']))
#test时只需要itemId和attri
test_data =pd.read_csv(r'intermediate_data/test_data.csv', usecols=['itemID', 'genre'])
test_data.drop_duplicates( keep='first', inplace=True)
test_data= np.array(test_data)


def construct_pipline():
    construt_train_arri_Emb()
    construt_negative_arri_Emb()
    construt_test_data()



def construt_negative_arri_Emb():
    for i in neg_data:
        i[2] = i[2][1:-1]

    for i in neg_data:
        tmp = np.linspace(0, 36, 19)  # 返回[0,34]内均匀的18个数字,0-34每两位表示是/否为该对应属性，用以look up G_generator_matrix
        li = np.int32(i[2].split(','))
        for j in li:
            j = j-1#因为atrri序号是1-19而不是0-18
            tmp[j] = tmp[j] + 1
        i[2] = np.array(tmp, dtype=np.int32)
    neg = pd.DataFrame(neg_data)
    neg.to_csv('final_neg_data.csv', header=None, index=0)


def construt_train_arri_Emb():
    for i in train_data:
        i[2] = i[2][1:-1]
    for i in train_data:
        tmp = np.linspace(0, 36, 19)
        li = np.int32(i[2].split(','))
        for j in li:
            j = j - 1  # 因为atrri序号是1-19而不是0-18
            tmp[j] = tmp[j] + 1
        i[2] = np.array(tmp, dtype=np.int32)
    train = pd.DataFrame(train_data)
    train.to_csv('final_train_data.csv', header=None, index=0)


def construt_test_data():
    for i in test_data:
        i[1] = i[1][1:-1]
    print(test_data)
    item_batch = [x[0] for x in test_data]
    attribute = []
    for i in test_data:
        tmp = np.linspace(0, 36, 19)
        li = np.int32(i[1].split(','))
        print(li)
        for j in li:
            j = j - 1  # 因为atrri序号是1-19而不是0-18
            tmp[j] = tmp[j] + 1
        attribute.append(tmp)
    item = pd.DataFrame(item_batch)
    item.to_csv('final_test_item.csv', header=None, index=0)
    attribute = pd.DataFrame(attribute)
    attribute.to_csv('final_test_attribute.csv', header=None, index=0)

# construct_pipline()