# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import csv
import numpy as np

path = '../data/info.csv'


def use_csv():
    x = []
    y = []
    num = 0
    with open(path, 'r') as csv_file:
        file_array = csv.reader(csv_file, delimiter=',')
        for line in file_array:
            if line[0].isdigit():
                x.append(line[5])
                y.append(line[6])
                num += 1
    # print '数据点总数：%d' % num
    return x, y


def use_numpy():
    x, y = np.loadtxt(path, delimiter=',', unpack=True, encoding='unicode')
    return x, y


def main():
    x, y = use_csv()
    # x, y = use_numpy()
    plt.scatter(x, y, label='ofo', c='b')
    plt.show()


if __name__ == '__main__':
    main()
