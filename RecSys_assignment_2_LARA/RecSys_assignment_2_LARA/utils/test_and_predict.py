# -*- coding: utf-8 -*-
'''
关于模型测试和预测的函数
'''


import numpy as np
import pandas as pd


user_emb_matrix = np.array(pd.read_csv(r'data/groud_user_attr.csv', header=None))#用于查询user_embedding
ui_matrix = np.array(pd.read_csv(r'data/ui_matrix.csv', header=None))
# user_attribute_matrix = np.array(pd.read_csv(r'util/user_attribute.csv',header=None)) #用于矩阵相乘计算余弦相似度

# test_data =pd.read_csv(r'data/final_test_data.csv',usecols=['itemID','genre'])
# test_data.drop_duplicates( keep='first', inplace=True)
# test_data= np.array(test_data)
test_item =np.array(pd.read_csv('data/final_test_item.csv', header =None).astype(np.int32))
test_attribute = np.array(pd.read_csv('data/final_test_attribute.csv', header =None).astype(np.int32))


#指标计算
def RR(r, k):
    for i in range(k):
        if r[i] == 1:
            return 1.0 / (i + 1.0)
    return 0

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])



##以下用于测试
def get_testdata():
    return test_item,test_attribute

#用于测试/推荐——————————和生成的用户相似的前k个用户
def get_intersection_similar_user(G_user, k):
#源码是用user_attribute_matrix（也就是未归一化的user_embedding）来计算相似度，但我改成了归一化后的user_embedding来计算相似度，试下效果
#     user_emb_matrixT = np.transpose(user_attribute_matrix)
    user_emb_matrixT = np.transpose(user_emb_matrix)
    A = np.matmul(G_user, user_emb_matrixT)
    intersection_rank_matrix = np.argsort(-A)
    return intersection_rank_matrix[:, 0:k]



# 测试 返回一堆指标
def test(test_item_batch, test_G_user):
    k_value = 20
    test_BATCH_SIZE = np.size(test_item_batch)

    test_intersection_similar_user = get_intersection_similar_user(test_G_user, k_value)

    count = 0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        for test_u in test_userlist:

            if ui_matrix[test_u, test_i] == 1:
                count = count + 1
    p_at_20 = round(count / (test_BATCH_SIZE * k_value), 4)

    ans = 0.0
    RS = []
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        r = []
        for user in test_userlist:
            r.append(ui_matrix[user][test_i])
        RS.append(r)
    #    print('MAP @ ',k_value,' is ',  evall.mean_average_precision(RS) )
    M_at_20 = mean_average_precision(RS)

    ans = 0.0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        r = []
        for user in test_userlist:
            r.append(ui_matrix[user][test_i])
        ans = ans + ndcg_at_k(r, k_value, method=1)
    #    print('ndcg @ ',k_value,' is ', ans/test_BATCH_SIZE)
    G_at_20 = ans / test_BATCH_SIZE
    k_value = 10

    count = 0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        for test_u in test_userlist[:k_value]:

            if ui_matrix[test_u, test_i] == 1:
                count = count + 1
    p_at_10 = round(count / (test_BATCH_SIZE * k_value), 4)

    ans = 0.0
    RS = []
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        r = []
        for user in test_userlist[:k_value]:
            r.append(ui_matrix[user][test_i])
        RS.append(r)
    #    print('MAP @ ',k_value,' is ',  evall.mean_average_precision(RS) )
    M_at_10 = mean_average_precision(RS)

    ans = 0.0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        r = []
        for user in test_userlist[:k_value]:
            r.append(ui_matrix[user][test_i])
        ans = ans + ndcg_at_k(r, k_value, method=1)
    #    print('ndcg @ ',k_value,' is ', ans/test_BATCH_SIZE)
    G_at_10 = ans / test_BATCH_SIZE

    return p_at_10, p_at_20, M_at_10, M_at_20, G_at_10, G_at_20

def RR(r, k):
    for i in range(k):
        if r[i] == 1:
            return 1.0 / (i + 1.0)
    return 0

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


