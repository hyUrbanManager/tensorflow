# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

light_file_path = '../data/spec_data.xlsx'
wave_file_path = '../data/wavelength.xlsx'
pic_save_path = '../pic/'

def read_light(row=0):
    data = pd.read_excel(light_file_path)
    light_data_frame = data.iloc[row, 7]
    light = light_data_frame.split(',')
    light_bg_data_frame = data.iloc[row, 8]
    light_bg = light_bg_data_frame.split(',')
    y = []
    for i in range(0, len(light)):
        y.append(int(light_bg[i]) - int(light[i]))

    return y


def read_wave_length():
    # 由于没有列名，所以额外指定名称wave。
    data = pd.read_excel(wave_file_path, header=None, names=['wave'])
    wave = data['wave']
    x = []
    for i in range(0, len(wave)):
        x.append(wave[i])

    return x


def main():
    x = read_wave_length()
    for index in range(0, 46):
        y = read_light(index)
        plt.plot(x, y, label='light', c='g')
        plt.savefig(pic_save_path + str(index) + '.png')
        plt.cla()


if __name__ == '__main__':
    main()
