from __future__ import print_function
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp

abs_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(abs_path, '..', 'toneclassifier', 'train')
test_path = os.path.join(abs_path, '..', 'toneclassifier', 'test_new')

count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

def plot_origin(cur_path):
    labels = open(os.path.join(cur_path, 'label.csv'), 'r')
    f0s = open(os.path.join(cur_path, 'f0.csv'), 'r')
    ens = open(os.path.join(cur_path, 'engy.csv'), 'r')
    content1 = labels.readlines()
    content2 = f0s.readlines()
    content3 = ens.readlines()
    for i in range(5):
        plt.figure(i)
        plt.ylim(-200, 200)
    for i in range(len(content1)):
        label = content1[i]
        f0 = content2[i]
        label = int(label)
        en = content3[i]
        if count[label] > 100:
            continue
        else:
            count[label] += 1
        plt.figure(label)
        f0 = f0.split(',')
        f0 = [float(f) for f in f0]
        f0 = np.array(f0)
        en = en.split(',')
        en = [float(e) for e in en]
        en = np.array(en)

        f0 = f0[np.where(en > 50.)[0]]
        f0 = f0[3:-3]
        f0 = f0[np.where(f0 > 50.)[0]]

        begin = 0
        dif = []
        for j in range(len(f0) - 1):
            dif.append(f0[j + 1] - f0[j])
        dif = np.array(dif)
        mean = np.abs(dif).mean()
        limit = 5.
        for j in range(len(dif)):
            if dif[j] > limit:
                dif[j] = limit
            elif dif[j] < -limit:
                dif[j] = -limit
        f0 = np.array(dif)


        x = np.linspace(0, 5, len(f0))

        #plt.plot(x, en)

        nx = np.linspace(0, 5, 99)
        fc = itp.interp1d(x, f0, kind='cubic')
        feature = fc(nx)
        f0 = [begin, ]
        for j in range(len(feature)):
            f0.append(begin + feature[j])
            begin += feature[j]
        nx = np.linspace(0, 5, 100)
        plt.plot(nx, f0)


    plt.show()

plot_origin(train_path)
