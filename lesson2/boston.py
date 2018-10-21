# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import sklearn 
from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# gain data.
dir(load_boston())
boston = load_boston()
x = boston.data
y = boston.target[:,np.newaxis]
print(boston.feature_names)
plt.scatter(x, y, label='ofo', c='b')
plt.show()

# create mode.which model
model = Sequential()
model.compile(optimizer=SGD(lr=0.01),loss='mse',metrics=['accuracy'])

print(pred)