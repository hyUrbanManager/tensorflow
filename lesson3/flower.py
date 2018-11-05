# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from skimage import io

# 训练图片文件夹路径。
data_path = 'G:\\ai\\Homework2\\flowers-recognition'

os.chdir(data_path)
img = io.imread("./flowers/rose/12240303_80d87f77a3_n.jpg")
print(img.shape)

plt.imshow(img)
plt.show()


# 选择前80%的图片用来训练
def divide_imgs(path):
    images_list = []
    labels_list = []
    class_list = []
    totalNum = 0

    for image in os.listdir(path):
        totalNum += 1
    trainNum = totalNum * 8 / 10

    num_count = 0
    for image in os.listdir(path):
        img_data = img.imread(os.path.join(path, image))
        images_list.append(img_data)
        # labels_list.append(folder)
        # class_list.extend([num_count])
        num_count += 1
        if num_count == n:
            break


        for images in os.listdir(img_path):
            img_data = img.imread(os.path.join(img_path, images))
            images_list.append(img_data)
            labels_list.append(folder)
            return images_list, labels_list, class_list
