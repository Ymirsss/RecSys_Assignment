# -*- coding: utf-8 -*-
'''
items number： 9724, 1945for test data set and 7779 for train data set.
用来从原始movielens数据集中合并数据、renumber数据、划分数据集、以及获取user_item交互矩阵
'''

import pandas as pd
import random
import numpy as np

'''
加载原始数据并对genre做数字化处理，产生entire_data.csv,下一步→→split_dataset()
'''
def LodaData():
    gen = ['Action','Adventure','Animation',"Children",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western','IMAX','(no genres listed)']
    # 评分
    rnames = ['userId', 'itemId', 'rating', 'timestamp']
    ratings = pd.read_csv('ml-latest-small/ratings.csv',
                          sep=',', header=None, names=rnames, engine='python')
    # 电影信息
    mnames = ['itemId', 'title', 'genre']
    movies = pd.read_csv('ml-latest-small/movies.csv',
                         sep=',', header=None, names=mnames, engine='python')
    temp1 = pd.merge(ratings,movies).drop(index=0)
    genre_li = []
    print(temp1)

    #编码atrri（按序）
    for genre_str in temp1['genre']:
        genre = genre_str.split('|')
        for j in range(len(genre)):
            genre[j] = gen.index(genre[j])
        genre_li.append(genre)

    temp1['genre'] = genre_li
    temp1.drop(['timestamp','title'],axis=1,inplace=True)
    items =  temp1['itemId'].reset_index(drop=True)
    items.drop_duplicates(keep='first', inplace=True)
    items = items.reset_index(drop=True)
    items = items.reset_index()
    print(' items is ', items)
    data = pd.merge(temp1, items, how='inner', on=['itemId'])
    # print(data)

    data.rename(columns={'index': 'itemID'}, inplace=True)
    data.to_csv('entire_data.csv', columns=['userId','itemID', 'rating', 'genre'], index=0)
    # print(temp1['itemId'].unique())
    # temp1.to_csv('entire_data.csv',columns=[ 'userId', 'itemId', 'rating', 'genre'],index=0)
# LodaData()


'''
基于entire_data.csv数据进一步划分train_data、neg_data、test_data
'''
def spilt_dataset():
    data = pd.read_csv('intermediate_data/entire_data.csv')

    pos_data = data[data['rating'] > 3]#打分4、5是正例用户
    neg_data = data[data['rating'] < 4]#打分1 2 3 是负例用户

    test_id = random.sample(range(9724), 1945)
    train_id = set(list(range(9724))) - set(test_id)
    test_id = pd.DataFrame(test_id)
    test_id.rename(columns={0: 'itemID'}, inplace=True)
    test_data = pd.merge(pos_data, test_id, how='inner', on=['itemID'])#merge:类似内连接 只保留pos_data, test_id在'itemId'上重合的部分，也就是说 只给test分配507个Item

    train_id = pd.DataFrame(list(train_id))
    train_id.rename(columns={0: 'itemID'}, inplace=True)
    train_data = pd.merge(pos_data, train_id, how='inner', on=['itemID'])#同上

    nega_data = pd.merge(neg_data, train_id, how='inner', on=['itemID'])

    test_data.to_csv('test_data.csv', index=0)
    train_data.to_csv('train_data.csv', index=0)
    nega_data.to_csv('neg_data.csv', index=0)

# spilt_dataset()

#获取user_item交互矩阵，用于指标计算
def get_ui_matrix():
    #源码是用的划分后的train_data生成的UI交互矩阵，我改成了entire_data，试下表现
    data = np.array(pd.read_csv(r'intermediate_data/entire_data.csv'))
    print(data)
    ui = np.zeros(shape=(610, 9742), dtype=np.int32)
    for i in data:
        ui[i[0]-1][i[1]] = 1
    save = pd.DataFrame(ui)
    print(save)
    save.to_csv('ui_matrix.csv', index=0, header=None)


get_ui_matrix()
