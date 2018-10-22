# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import sklearn 
from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from sklearn import linear_model

# gain data.
dir(load_boston())
boston = load_boston()
x = boston.data
y = boston.target[:,np.newaxis]
# train
x_train = x[:450]
y_train = y[:450]
# pred
x_test = x[450:]
y_test = y[450:]

# 卷积网络模型。
model = Sequential()
model.add(Dense(units=64, input_dim=13))
model.add(Activation("relu"))
model.add(Dense(units=1))
model.add(Activation("softmax"))
model.compile(optimizer=SGD(lr=0.01),loss='mse',metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=6, batch_size=32)
# pred = model.predict(x_test, batch_size=128)

# sklearn默认的线性模型。
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)

plt.scatter(pred, y_test, label='boston', c='b')
plt.xlabel('pred')
plt.ylabel('truth')
plt.grid()
plt.title('xld 15210320411')
plt.show()