# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import test_and_predict
from utils import get_batch_data

#超参
alpha = 0  # 正则项参数
attr_num = 19
attr_present_dim = 5#生成器将每个属性转化为attr_present_dim 维
batch_size = 1024
hidden_dim = 100
user_emb_dim = attr_num
learning_rate = 0.0001
epoch = 400

def glorot_init(shape):
    return tf.random_normal(shape=shape,stddev=1./tf.sqrt(shape[0]/2.0))
#权重与偏移
weights = {
    # 'D_matrix' :tf.Variable([2*attr_num, attr_present_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'D_W1':tf.Variable([attr_num*attr_present_dim  + user_emb_dim , hidden_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'D_W2':tf.Variable([hidden_dim, hidden_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'D_W3':tf.Variable([hidden_dim, user_emb_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'G_matrix':tf.Variable([2*attr_num, attr_present_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'G_W1':tf.Variable([attr_num*attr_present_dim , hidden_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'G_W2':tf.Variable([hidden_dim, hidden_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'G_W3':tf.Variable([hidden_dim, user_emb_dim],initializer=tf.contrib.layers.xavier_initializer())
    'D_matrix' :tf.Variable(glorot_init([2*attr_num, attr_present_dim])),
    'D_W1':tf.Variable(glorot_init([attr_num*attr_present_dim  + user_emb_dim , hidden_dim])),
    'D_W2':tf.Variable(glorot_init([hidden_dim, hidden_dim])),
    'D_W3':tf.Variable(glorot_init([hidden_dim, user_emb_dim])),
    'G_matrix':tf.Variable(glorot_init([2*attr_num, attr_present_dim])),
    'G_W1':tf.Variable(glorot_init([attr_num*attr_present_dim , hidden_dim])),
    'G_W2':tf.Variable(glorot_init([hidden_dim, hidden_dim])),
    'G_W3':tf.Variable(glorot_init([hidden_dim, user_emb_dim]))

}
bias={
    # 'D_b1':tf.Variable([1, hidden_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'D_b2':tf.Variable([1, hidden_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'D_b3':tf.Variable([1, user_emb_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'G_b1':tf.Variable([1, hidden_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'G_b2':tf.Variable([1, hidden_dim],initializer=tf.contrib.layers.xavier_initializer()),
    # 'G_b3':tf.Variable([1, user_emb_dim],initializer=tf.contrib.layers.xavier_initializer())
    'D_b1':tf.Variable(glorot_init([1, hidden_dim])),
    'D_b2':tf.Variable(glorot_init([1, hidden_dim])),
    'D_b3':tf.Variable(glorot_init([1, user_emb_dim])),
    'G_b1':tf.Variable(glorot_init([1, hidden_dim])),
    'G_b2':tf.Variable(glorot_init([1, hidden_dim])),
    'G_b3':tf.Variable(glorot_init([1, user_emb_dim]))
}
#训练输入
#此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
attribute_arr  = tf.placeholder(shape=[None,attr_num],dtype = tf.int32)
real_user_emb = tf.placeholder(shape = [None, user_emb_dim], dtype = tf.float32)
neg_attribute_arr  = tf.placeholder(shape=[None,attr_num],dtype = tf.int32)
neg_user_emb = tf.placeholder(shape = [None, user_emb_dim], dtype = tf.float32)


#构建生成器和判别器
def Generator(attribute_arr):#一批一批传进来
    attri_present = tf.nn.embedding_lookup(weights['G_matrix'],attribute_arr)
    attri_feature = tf.reshape(attri_present,shape=[-1,attr_num*attr_present_dim])
    l1_output = tf.nn.tanh(tf.matmul( attri_feature ,weights['G_W1']) + bias['G_b1'])
    l2_output = tf.nn.tanh(tf.matmul(l1_output, weights['G_W2']) + bias['G_b2'])
    gen_user = tf.nn.tanh(tf.matmul(l2_output, weights['G_W3']) + bias['G_b3'])

    return gen_user

def Discriminator(attribute_arr,user_emb):  # 一批一批传进来
    attri_present = tf.nn.embedding_lookup(weights['D_matrix'], attribute_arr)
    attri_feature = tf.reshape(attri_present, shape=[-1, attr_num * attr_present_dim])
    fed_emb = tf.concat([attri_feature, user_emb],1)

    l1_output = tf.nn.tanh(tf.matmul(fed_emb, weights['D_W1']) + bias['D_b1'])
    l2_output = tf.nn.tanh(tf.matmul(l1_output, weights['D_W2']) + bias['D_b2'])
    D_logit = tf.matmul(l2_output, weights['D_W3']) + bias['D_b3']
    D_prob = tf.nn.tanh(D_logit)

    return D_prob,D_logit


#生成
gen_user_emb = Generator(attribute_arr )
#判别(三种用户对)
D_real, D_logit_real = Discriminator(attribute_arr , real_user_emb)
D_gen, D_logit_gen = Discriminator(attribute_arr , gen_user_emb)
D_neg, D_logit_neg = Discriminator(neg_attribute_arr , neg_user_emb)

#建立损失(二分类交叉熵)
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_gen, labels=tf.zeros_like(D_logit_gen)))
D_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_neg, labels=tf.zeros_like(D_logit_neg)))

#模型复杂度的regular
D_regular = alpha * (tf.nn.l2_loss(weights['D_matrix'])  + tf.nn.l2_loss(weights['D_W1']) + tf.nn.l2_loss(bias['D_b1']) + tf.nn.l2_loss(weights['D_W2']) + tf.nn.l2_loss(bias['D_b2']) + tf.nn.l2_loss(weights['D_W3']) + tf.nn.l2_loss(bias['D_b3']))
G_regular = alpha * (tf.nn.l2_loss(weights['G_matrix'])  + tf.nn.l2_loss(weights['G_W1']) + tf.nn.l2_loss(bias['G_b1']) + tf.nn.l2_loss(weights['G_W2']) + tf.nn.l2_loss(bias['G_b2']) + tf.nn.l2_loss(weights['G_W3']) + tf.nn.l2_loss(bias['G_b3']))


D_loss = (1-alpha)*(D_loss_real + D_loss_gen  + D_loss_neg)+ D_regular
G_loss = (1-alpha)*(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_gen, labels=tf.ones_like(D_logit_gen)))) + G_regular


#创建优化器
#tensorflow梯度下降过程默认更新所有参数，于是这里手动设置更新的参数
D_param = [weights['D_matrix'],weights['D_W1'],bias['D_b1'],weights['D_W2'],bias['D_b2'],weights['D_W3'],bias['D_b3']]
G_param = [weights['G_matrix'],weights['G_W1'],bias['G_b1'],weights['G_W2'],bias['G_b2'],weights['G_W3'],bias['G_b3']]

D_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(D_loss, var_list=D_param)
G_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(G_loss, var_list=G_param)

#初始化变量
init = tf.global_variables_initializer()
#setting
config = tf.ConfigProto()#在创建会话的时候进行参数配置。
config.gpu_options.allow_growth = True#开始分配少量显存，然后按需增加
sess = tf.Session(config=config)
sess.run(init)

#测试
#要存储的结果
p_10_to_save = []
p_20_to_save = []
M_10_to_save = []
M_20_to_save = []
G_10_to_save = []
G_20_to_save = []
#用于测试
global max_p_at_10
global max_p_at_20
global max_M_at_10
global max_M_at_20
global max_G_at_10
global max_G_at_20
global p_at_10
global p_at_20
global M_at_10
global M_at_20
global G_at_10
global G_at_20
global g_loss_li
global d_loss_li
# 训练
def train():
    max_p_at_10 = 0
    max_p_at_20 = 0
    max_M_at_10 = 0
    max_M_at_20 = 0
    max_G_at_10 = 0
    max_G_at_20 = 0
    g_loss_li = []
    d_loss_li = []
    start = time.time()
    for epo in tqdm(range(epoch)):
        get_batch_data.shuffle()
        index = 0
        while index < 41463:  # neg_data contains 41463 rows，小于train data 所以要以小的为限制
            if index + batch_size <= 41463:
                train_user_batch, train_item_batch, train_attr_batch, train_user_emb_batch = get_batch_data.get_traindata(index, index + batch_size)
                neg_user_batch, neg_item_batch, neg_attr_batch, neg_user_emb_batch = get_batch_data.get_negdata(index, index + batch_size)
            index = index + batch_size
            #生成器
            _, d_loss, gen_usr = sess.run([D_optimizer, D_loss, gen_user_emb],#sess.run对应输出[D_optimizer, D_loss, gen_user_emb
                                              feed_dict={attribute_arr: train_attr_batch,
                                                         real_user_emb: train_user_emb_batch,
                                                         neg_attribute_arr: neg_attr_batch,
                                                         neg_user_emb: neg_user_emb_batch
                                                         })

            #判别器
            _, g_loss = sess.run([G_optimizer, G_loss], feed_dict={attribute_arr: train_attr_batch})

        end = time.time()
            # print("\n Epoch:%d,time:%.3fs, d_loss:%.4f, g_loss:%.4f " % (epo,(end - start), d_loss, g_loss))

    #每10次迭代测试一次
        print("\n Epoch:%d,time:%.3fs, d_loss:%.4f, g_loss:%.4f " % (epo, (end - start), d_loss, g_loss))
        g_loss_li.append(g_loss)
        d_loss_li.append(d_loss)
        test_item_batch, test_attribute_vec = test_and_predict.get_testdata()
        test_G_user = sess.run(gen_user_emb, feed_dict={attribute_arr: test_attribute_vec})
        #        print( test_G_user[:10])
        p_at_10, p_at_20, M_at_10, M_at_20, G_at_10, G_at_20 = test_and_predict.test(test_item_batch, test_G_user)
        if p_at_10 > max_p_at_10:
            max_p_at_10 = p_at_10
        p_10_to_save.append(p_at_10)
        if p_at_20 > max_p_at_20:
            max_p_at_20 = p_at_20
        p_20_to_save.append(p_at_20)
        if M_at_10 > max_M_at_10:
            max_M_at_10 = M_at_10
        M_10_to_save.append(M_at_10)
        if M_at_20 > max_M_at_20:
            max_M_at_20 = M_at_20
        M_20_to_save.append(M_at_20)
        if G_at_10 > max_G_at_10:
            max_G_at_10 = G_at_10
        G_10_to_save.append(G_at_10)
        if G_at_20 > max_G_at_20:
            max_G_at_20 = G_at_20
        G_20_to_save.append(G_at_20)

        print('p_at_10 ', p_at_10, 'p_at_20', p_at_20, 'M_at_10', M_at_10, 'M_at_20',
              M_at_20, 'G_at_10', G_at_10, 'G_at_20', G_at_20)
        print('max p_at_10 ', max_p_at_10, 'p_at_20', max_p_at_20, 'M_at_10', max_M_at_10, 'M_at_20',
              max_M_at_20, 'G_at_10', max_G_at_10, 'G_at_20', max_G_at_20)
        if (epo % 100 == 0) and (epo != 0):
            pd.DataFrame(p_10_to_save).to_csv('result/p10.csv')
            pd.DataFrame(p_20_to_save).to_csv('result/p20.csv')
            pd.DataFrame(M_10_to_save).to_csv('result/m10.csv')
            pd.DataFrame(M_20_to_save).to_csv('result/m20.csv')
            pd.DataFrame(G_10_to_save).to_csv('result/g10.csv')
            pd.DataFrame(G_20_to_save).to_csv('result/g20.csv')
            pd.DataFrame(g_loss_li).to_csv('result/gLoss.csv')
            pd.DataFrame(d_loss_li).to_csv('result/dloss.csv')

def pplot():
    p_at_10 = pd.read_csv(r'result/p10.csv', header=None)

    plt.subplot(211)
    plt.plot(range(int(epoch/10)), p_at_10, marker='o', label='Test Data')
    plt.title('The MovieLens Test Data')
    plt.xlabel('Number of Epochs')
    plt.ylabel('p_at_10')
    plt.legend()
    plt.grid()
    plt.show()

    # plt.subplot(212)
    # plt.plot(range(epoch), d_loss_li, marker='o', label='d_loss')
    # plt.plot(range(epoch), g_loss_li, marker='v', label='g_loss')
    # plt.title('The MovieLens Test Data')
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid()
    # plt.show()

train()
pplot()
