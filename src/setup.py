import numpy as np
import random
import os
import scipy.interpolate as itp

def pick_nonzero(feature):
    indice = np.where(feature > 0)
    if indice[0][-1] - indice[0][0] + 1 == len(indice[0]):
        return feature[indice]
    else:
        pos = [i for i in range(indice[0][0], indice[0][-1] + 1)]
        return feature[pos]


def exclude_noise(feature, exclude_st):
    if exclude_st < 0:
        return feature
    feature = np.array(feature)
    limit = exclude_st
    tmp = []
    for i in range(feature.shape[0]):
        if feature[i] < -limit:
            pass
            tmp.append(-limit)
        elif feature[i] > limit:
            pass
            tmp.append(limit)
        else:
            tmp.append(feature[i])
    return np.array(tmp)


def pre_processf0(feature, New_length, exclude_st):
    feature = pick_nonzero(feature)
    tmp = []
    for j in range(feature.shape[0] - 1):
        tmp.append(feature[j+1] - feature[j])
    feature = exclude_noise(np.array(tmp), exclude_st)
    curx = np.linspace(0, New_length, feature.shape[0])
    newx = np.linspace(0, New_length, New_length)
    fc = itp.interp1d(curx, feature, kind='cubic')
    feature = fc(newx)
    return feature


def exclude_by_engy(en, feature, threshold, cut_end):
    feature = feature[np.where(en > threshold)[0]]
    feature = feature[cut_end:-cut_end]
    return feature


def exclude_by_f0(feature, threshold):
    feature = feature[np.where(feature > threshold)]
    return feature


def prepare(path, config):
    labeldct = {'one': 0, 'two': 1, 'three': 2, 'four': 3}
    A = []
    B = []
    for suf in ['one', 'two', 'three', 'four']:
        filelist = os.listdir(os.path.join(path, suf))
        filelist.sort()
        for i in range(0, len(filelist), 2):
            A.append(labeldct[suf])

            f0_out = []
            en_out = []

            f0 = open(path + '/' + suf + '/' + filelist[i + 1], 'r')
            f0_data = f0.readlines()
            for j in range(len(f0_data)):
                f0_out.append(float(f0_data[j].split('\n')[0]))

            engy = open(path + '/' + suf + '/' + filelist[i], 'r')
            engy_data = engy.readlines()
            for j in range(len(engy_data)):
                en_out.append(float(engy_data[j].split('\n')[0]))
            B.append(f0_out + en_out)

            f0.close()
            engy.close()

    ret1 = []
    ret2 = []
    for i in range(len(A)):
        cur_sz = len(B[i]) >> 1
        tmp1 = np.array(B[i][0:cur_sz], dtype='float64')
        tmp2 = np.array(B[i][cur_sz:], dtype='float64')

        ret1.append(A[i])
        tmp1 = exclude_by_engy(tmp2, tmp1, config['threshold_engy'], config['cut_end'])
        tmp1 = exclude_by_f0(tmp1, config['threshold_f0'])
        cut1 = pre_processf0(tmp1, config['New_length'], config['exclude_st'])

        tmp = list(cut1)

        ret2.append(tmp)

    ret1 = [int(s) for s in ret1]
    X = [i for i in range(len(ret1))]
    random.seed(23333)
    random.shuffle(X)
    random.shuffle(X)
    random.shuffle(X)
    A = []
    B = []
    for i in range(len(ret1)):
        A.append(ret1[X[i]])
        B.append(ret2[X[i]])
    ret1 = A
    ret2 = B

    label_file = open(path + '/labelnew.csv', 'wb')
    f0_file = open(path + '/datanew.csv', 'wb')
    for num in ret1:
        label_file.write(bytes(str(num) + '\n', 'UTF-8'))
    for num in ret2:
        for j in num[:-1]:
            f0_file.write(bytes(str(j) + ',', 'UTF-8'))
        f0_file.write(bytes(str(num[-1]) + '\n', 'UTF-8'))
    return ret1, ret2
