# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

'''  use interaction and item attributes to build user presentation   '''


def user_emb():
    data = pd.read_csv(r'intermediate_data/train_data.csv', usecols=['userId', 'genre'])
    #    print(data)

    data['tmp'] = data['genre'].str.split('[', expand=True)[1]
    data['tmp1'] = data['tmp'].str.split(']', expand=True)[0]##得到了attr列表

    #    print(data)
    user = np.array(data['userId'])
    attr = np.array(data['tmp1'])

    print(len(user))
    print(len(attr))
    #
    #    print(attr[109800])
    #    attr_list=np.int32(attr[109800].split(','))
    #    print(type(attr_list[1]))
    user_present = np.zeros(shape=(610, 19), dtype=np.int32)

    for i in range(len(user)):
        attr_list = np.int32(attr[i].split(','))
        for j in attr_list:
            '''
           统计所有用户评分过的item的类别频率
            '''
            user_present[user[i]-1][j-1] += 1.0#因为userId比实际索引大1

    save = pd.DataFrame(user_present)
    #未归一化的user_embedding
    save.to_csv('groud_user_attr.csv', index=0, header=None)

    save['Col_sum'] = save.apply(lambda x: x.sum(), axis=1)  # 在最后一行添加每一行总和，用于下面的归一化
    save = np.array(save, dtype=np.float32)
    print(save)
    for i in range(610):
        tt = save[i][-1]

        if tt != 0.0:
            for j in range(19):
                save[i][j] = save[i][j] / tt  # 将user_present归一化

# 做归一化处理的原因：如果一个用户很活跃，给很多片子都点了喜欢，那么他在某几个类型上必然值很大，但是如果这几个atri都高，则没有表示对类型的偏好信息，只是因为user评分过的电影多而已，归一化之后才能看出比例
    #也是为了让所有用户的embedding在同一个量级 而不是一个是都是几十 一个都是个位

    save = pd.DataFrame(save)
    print(save)
    save.to_csv('groud_user_emb.csv', index=0, header=None,
                columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18])


user_emb()
