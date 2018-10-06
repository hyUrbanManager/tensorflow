# -*- coding:utf-8 -*-
##required packages
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD

###create data and add some noise
#####如何用numpy建立一个等差数列，并加入噪声(噪声服从高斯分布)
x_=np.linspace(-1,1,100)[:,np.newaxis]####x_.shape=(100,1)
y_=3*x_+1+np.random.standard_normal(x_.shape)*0.2####(mean=0, stdev=1)
#####也可以这么写####
'''
x_=np.linspace(-1,1,100)
x_=np.expand_dims(x_, axis=0)
y_=3*x+np.random.randn(x.shape[1])*0.2+1
'''