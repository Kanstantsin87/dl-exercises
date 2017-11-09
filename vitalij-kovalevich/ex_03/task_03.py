%matplotlib inline
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import numpy.linalg 
import numpy as np 
import re

from numpy import genfromtxt
file = open('data.txt')
file = file.readlines()
raw_data  = np.zeros([60, 17])
for i in range(60):
    result = re.split('\s+', file[i])
    for k in range(17):
        raw_data[i][k] = float(result[k])
X_all = raw_data[:,0:16]
Y_all = raw_data[:,16:]

indexes = np.arange(0, 48, 1)

np.random.seed(31)
np.random.shuffle(indexes)


X_train, X_test = X_all[indexes[0:48],:],X_all[indexes[48:60],:]
y_train, y_test = Y_all[indexes[0:48],:],Y_all[indexes[48:60],:]


p = pinv(X_train.T.dot(X_train)).dot(X_train.T)
w = p.dot(y_train)


def mse(y_pred, y):
    return (y_pred-y).T.dot(y_pred-y)/len(y_pred)
y_pred = w.T.dot(X_train.T)



p_test = pinv(X_test.T.dot(X_test)).dot(X_test.T)
w_test = p_test.dot(y_test)
y_test_pred = w_test.T.dot(X_test.T)

plt.figure(figsize=(16, 8))
plt.scatter(X_train[:, 0], y_train, c='b')   
plt.scatter(X_train[:, 0], y_pred, c='r') 
plt.show()


