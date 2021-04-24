import matplotlib.pyplot as plt
import numpy as np
import random

from getData import getData
from sklearn.model_selection import train_test_split
from PMF_Model import PMF_Model


if __name__ == "__main__":
    #获取数据及数据集划分
    file = "data/ml-latest-small/ratings.csv"
    data = getData(file)
    train, test = train_test_split(data, test_size=0.2)
    M =int(max(np.amax(train[:, 0]), np.amax(test[:, 0]))) + 1  # 第0列，user总数
    N =int(max(np.amax(train[:, 1]), np.amax(test[:, 1]))) + 1
    print("M: %d, N: %d" % (M,N))
    #PMF
    model = PMF_Model()
    model.train(train,test,M,N,k=10)

    #plot
    print("precision_acc,recall_acc:" + str(model.Metric(test)))
    plt.subplot(211)
    plt.plot(range(model.max_epoch), model.train_rmse_li, marker='o', label='Training Data')
    plt.plot(range(model.max_epoch), model.test_rmse_li, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()

    plt.subplot(212)
    plt.plot(range(model.max_epoch),model.loss_li,marker='o',label="Training Loss")
    plt.title('The MovieLens Dataset Loss Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()




def spilt_rating_dat(data, size=0.2):
    train_data = []
    test_data = []
    for line in data:
        rand = random.random()
        if rand < size:
            test_data.append(line)
        else:
            train_data.append(line)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    print('train length: %d \n test length: %d' % (len(train_data), len(test_data)))
    return train_data, test_data