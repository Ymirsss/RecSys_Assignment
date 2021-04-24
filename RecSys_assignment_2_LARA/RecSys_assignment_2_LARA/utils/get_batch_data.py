# -*- coding: utf-8 -*-
'''
获取batch数据
'''

import numpy as np
import pandas as pd


train = np.array(pd.read_csv('data/final_train_data.csv', header=None))
user_emb_matrix = np.array(pd.read_csv(r'data/groud_user_emb.csv', header=None))
neg = np.array(pd.read_csv(r'data/final_neg_data.csv', header=None))

def shuffle():
    np.random.shuffle(train)
    np.random.shuffle(neg)

# 获取batch train data
def get_traindata(start_index, end_index):
    batch_data = train[start_index: end_index]

    #    print(  batch_data)
    user_batch = [x[0]-1 for x in batch_data]#batch个user
    item_batch = [x[1] for x in batch_data]
    attr_batch = [x[2][1:-1].split() for x in batch_data] #list of list
    real_user_emb_batch = user_emb_matrix[user_batch]

    return user_batch, item_batch, attr_batch, real_user_emb_batch




def get_negdata(start_index, end_index):
    '''get negative samples'''
    batch_data = neg[start_index: end_index]

    #    print(  batch_data)
    user_batch = [x[0]-1 for x in batch_data]
    item_batch = [x[1] for x in batch_data]
    attr_batch = [x[2][1:-1].split() for x in batch_data]
    neg_user_emb_batch = user_emb_matrix[user_batch]  ###neg_userd的embedding是从user_embedding中查到的

    return user_batch, item_batch, attr_batch, neg_user_emb_batch