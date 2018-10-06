# -*- coding:utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 解决中文乱码问题
# sans-serif就是无衬线字体，是一种通用字体族。
# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica, 中文的幼圆、隶书等等。
# 指定默认字体 SimHei为黑体。
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

xs = range(0, 10, 1)
ys = []
xs2 = range(0, 10, 1)
ys2 = []
for x in xs:
    ys.append(x * x)
    ys2.append(x * x * x)

# plt.plot(xs, ys, label='x平方')
# plt.plot(xs2, ys2, label='x立方')
# plt.plot(xs, ys, label = 'x2')
# plt.plot(xs2, ys2, label = 'x3')
# plt.plot(xs, ys)
# plt.plot(xs2, ys2)

plt.scatter(xs, ys, label='xx', s=20, c='b', marker='o')
plt.scatter(xs2, ys2)

plt.xlabel("x")
plt.ylabel("y")
# plt.title("数学图形")
plt.legend()
plt.show()
