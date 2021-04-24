import numpy as np

class PMF_Model():

    def __init__(self,lr=0.8,lambda_u=0.1,lambda_i=0.1,max_batch=10,batch_size=1000,max_epoch=10):
        # self.k = k #latent dimension
        self.lr = lr #学习率
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i #正则化
        self.max_batch = max_batch #batch数
        self.batch_size = batch_size
        self.max_epoch = max_epoch #迭代次数
        self.U = None
        self.V = None
        self.train_rmse_li = []
        self.test_rmse_li = []
        self.loss_li = []


    def train(self,train,test,M,N,k):
       epoch = 0
       loss = 0
       pre_rmse = 1
       self.U = np.random.normal(0,0.1,(M,k))
       self.V = np.random.normal(0, 0.1, (N, k))
       print(self.U)
       mean_score = np.mean(train[:,2])

       while epoch < self.max_epoch:
          epoch += 1
          train_rmse_temp = 0
          test_rmse_temp = 0
          train_rmse = 0
          test_rmse = 0
          loss_temp = 0
          loss = 0
          gradient_U = 0
          gradient_V = 0

          #每次迭代打乱数据
          np.random.shuffle(train)

          for batch in range(self.max_batch):
            error = 0
            self.lr = 0.9*self.lr
            # print("epoch %d batch %d " % (epoch, batch + 1))
            batch_data = train[batch*self.batch_size:(batch+1)*self.batch_size]

#以下是mini-batch梯度下降法，实验验证，并不如下面常规sgd方法效果好
            # for line in batch_data:
            #     u_id,i_id,gold_rating = line
            #     u_id,i_id,gold_rating = int(u_id),int(i_id),float(gold_rating)
            #     pred_rating = np.dot(self.U[u_id],self.V[i_id].T)
            #     if ((pred_rating > 5) or (pred_rating < 0)):
            #         pred_rating = mean_score
            #     error = pred_rating-gold_rating+mean_score
            #     gradient_U += error * self.V[i_id] - self.lambda_u * self.U[u_id]
            #     gradient_V += error * self.U[u_id] - self.lambda_i * self.V[i_id]
            # gradient_U = gradient_U/self.batch_size
            # gradient_V = gradient_U/self.batch_size
            # for i in range(len(gradient_U)):
            #     if (abs(gradient_U[i]) < 0.001 ) :
            #         gradient_U[i] = 0
            #     if (abs(gradient_V[i]) < 0.001 ):
            #         # print("??????????????????????????????????")
            #         gradient_V[i] = 0
            # # print(gradient_U,gradient_V)
            # #计算梯度并更新
            # for line in batch_data:
            #     u_id,i_id,gold_rating = line
            #     u_id, i_id, gold_rating = int(u_id), int(i_id), float(gold_rating)
            #     self.U[u_id] += self.lr*gradient_U
            #     self.V[i_id] += self.lr*gradient_V
            # print(self.U)
#sgd 分批只是为了每次计算量小一点
            for line in batch_data:
                  u_id,i_id,gold_rating = line
                  u_id,i_id,gold_rating = int(u_id),int(i_id),float(gold_rating)
                  pred_rating = np.dot(self.U[u_id],self.V[i_id].T)
                  if((pred_rating > 5) or (pred_rating < 0)):
                      pred_rating = mean_score
                  error = pred_rating-gold_rating+mean_score
                  gradient_u = error*self.V[i_id]-self.lambda_u*self.U[u_id]
                  gradient_v = error*self.U[u_id] - self.lambda_i * self.V[i_id]
                  for i in range(len(gradient_u)):
                      if (abs(gradient_u[i]) < 0.001):
                          gradient_u[i] = 0
                      if (abs(gradient_v[i]) < 0.001):
                          gradient_v[i] = 0
                      if (abs(gradient_u[i]) > 1):
                          gradient_u[i] = 0
                      if (abs(gradient_v[i]) > 1):
                          gradient_v[i] = 0
                  self.U[u_id] += self.lr * gradient_u
                  self.V[i_id] += self.lr * gradient_v
                  # print(gradient_u,gradient_v)

        #每一迭代更新完成后，计算训练集mse和loss
          for line in train:
              u_id, i_id, gold_rating = line
              u_id, i_id, gold_rating = int(u_id), int(i_id), float(gold_rating)
              pred_rating = np.dot(self.U[u_id], self.V[i_id].T)
              train_rmse_temp += np.square(pred_rating-gold_rating+mean_score )
              loss_temp += self.lambda_u*np.dot(self.U[u_id],self.U[u_id].T)+self.lambda_i*np.dot(self.V[i_id],self.V[i_id].T)
          train_rmse = np.sqrt(train_rmse_temp/len(train))
          loss = 0.5*(train_rmse_temp+loss_temp)
          self.train_rmse_li.append(train_rmse)
          self.loss_li.append(loss)

         #计算每一个迭代测试集rmse
          for line in test:
              u_id, i_id, gold_rating = line
              u_id, i_id, gold_rating = int(u_id), int(i_id), float(gold_rating)
              pred_rating = np.dot(self.U[u_id], self.V[i_id].T)
              # if ((pred_rating > 5) or (pred_rating < 0)):
              #     pred_rating = mean_score
              test_rmse_temp += np.square(pred_rating - gold_rating+mean_score)
          test_rmse = np.sqrt(test_rmse_temp/len(test))
          self.test_rmse_li.append(test_rmse)
          print('Training RMSE: %f, Traning Loss:%f, Test RMSE %f' % (train_rmse,loss, test_rmse))

        #迭代停止条件
          if abs(train_rmse - pre_rmse)<0.0000000000001:
              epoch = self.max_epoch
          pre_rmse = train_rmse


    def predict_items(self,user):
        return  np.dot(self.U[user],self.V.T)



    #评估模型
    def Metric(self,test,k=3):
        user_li = np.unique(test[:, 0])
        pred = {}
        for user in user_li:
            if pred.get(user, None) is None:
                # print(self.predict(int(user)))
                pred[user] = np.argsort(self.predict_items(int(user)))[-k:]  # numpy.argsort索引排序
                print("用户%d的top3推荐如下："% user)
                for item in pred[user]:
                    print("Movie:%d,评分:%f"%(int(item),np.dot(self.U[int(user)],self.V[int(item)].T)))


        hit_k_cnt = {}
        for i in range(test.shape[0]):
            if test[i, 1] in pred[test[i, 0]]:
                hit_k_cnt[test[i, 0]] = hit_k_cnt.get(test[i, 0], 0) + 1
        item_cnt_per_user = np.bincount(np.array(test[:, 0], dtype='int32'))#test里每个user评分过的item数user

        precision_acc = 0.0
        recall_acc = 0.0
        for user in user_li:
            precision_acc += hit_k_cnt.get(user, 0) / float(k)#击中准确率
            recall_acc += hit_k_cnt.get(user, 0) / float(item_cnt_per_user[int(user)])
        print("hit@10 for users:",hit_k_cnt.items())
        return precision_acc / len(user_li), recall_acc / len(user_li)
