from __future__ import print_function
import os
import random
import numpy as np
import cntk as C
from cntk.learner import sgd, learning_rate_schedule, UnitType
from cntk.layers import Dense
from cntk.layers import Convolution as Conv
from cntk.layers import MaxPooling as MaxPool
from cntk.layers import AveragePooling as AvgPool

random.seed(22222)

RESIZE = 100
channels = 2
en_norm_st = 500
f0_norm_st = 2000
IntervalNum = 5
init_scale = 1
Counter = {'train': 0, 'test': 0}

def makeflag(arr, limit):
    flag = []
    for i in range(len(arr)):
        flag.append(1)
    for j in range(len(arr)):
        if arr[j] > limit['down'] and arr[j] < limit['up']:
            break
        else:
            flag[j] = 0
    for j in range(len(arr)):
        if arr[len(arr) - j - 1] > limit['down'] and arr[len(arr) - j - 1] < limit['up']:
            break
        else:
            flag[len(arr) - j - 1] = 0
    return flag


def pick_nonzero(feature):
    indice = np.where(feature > 0)
    if indice[0][-1] - indice[0][0] + 1 == len(indice[0]):
        return feature[indice]
    else:
        pos = [i for i in range(indice[0][0], indice[0][-1] + 1)]
        return feature[pos]

def pre_processf0(feature):
    feature = pick_nonzero(feature)
    max_key = np.max(feature)
    cur = []
    for f0 in feature:
        cur.append(f0 / max_key * f0_norm_st)
    feature = cur
    cur = []
    mean = np.array(feature).mean()
    for f0 in feature:
        cur.append(f0 - mean)
    feature = []
    for i in range(RESIZE):
        po = 1. * i / RESIZE * (len(cur) - 1)
        p1 = int(po)
        p2 = p1 + 1
        feature.append(cur[p1] * (po - p1) + cur[p2] * (p2 - po))
    return np.array(feature)


def pre_processen(feature):
    feature = pick_nonzero(feature)
    cur = []
    for en in feature:
        cur.append(en * en * en)
    max_key = np.array(cur).max()
    feature = []
    for en in cur:
       feature.append(en / max_key * en_norm_st)
    cur = []
    for i in range(RESIZE):
        po = 1. * i / RESIZE * (len(feature) - 1)
        p1 = int(po)
        p2 = p1 + 1
        cur.append(feature[p1] * (po - p1) + feature[p2] * (p2 - po))
    return np.array(cur)


def prepare(path):
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

    X = [i for i in range(len(A))]
    for i in range(len(A)):
        random.shuffle(X)

    ret1 = []
    ret2 = []
    for i in range(len(A)):
        ret1.append(A[X[i]])
        cursz = len(B[X[i]]) >> 1
        tmp1 = np.array(B[X[i]][0:cursz], dtype='float64')
        tmp2 = np.array(B[X[i]][cursz:], dtype='float64')
        tmp1 = pre_processf0(tmp1)
        tmp2 = pre_processen(tmp2)
        tmp = list(tmp1) + list(tmp2)
        ret2.append(tmp)

    ret1 = [int(s) for s in ret1]
    '''
    label_file = open(path + '/labelnew.csv', 'wb')
    f0_file = open(path + '/datanew.csv', 'wb')
    for num in ret1:
        label_file.write(bytes(str(num) + '\n', 'UTF-8'))
    for num in ret2:
        for j in num[:-1]:
            f0_file.write(bytes(str(j) + ',', 'UTF-8'))
        f0_file.write(bytes(str(num[-1]) + '\n', 'UTF-8'))
    '''
    return ret1, ret2

abs_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(abs_path, '..', 'toneclassifier', 'train')
test_path = os.path.join(abs_path, '..', 'toneclassifier', 'test_new')
train_label, train_data = prepare(train_path)
test_label, test_data = prepare(test_path)

def generate_data(sample_size, data, labels, type):
    topick = []
    for i in range(sample_size):
        topick.append(Counter[type])
        Counter[type] = (Counter[type] + 1) % len(labels)
    Y = []
    for i in topick:
        tmp = []
        for j in range(4):
            tmp.append(labels[i] == j)
        Y.append(tmp)
    Y = np.array(Y, dtype=np.float32)
    X = np.array([[data[i]] for i in topick])
    X = X.reshape((sample_size, channels, -1, 1))
    return X, Y


def CNN():

    input = C.ops.input_variable((2, RESIZE, 1), np.float32)
    label = C.ops.input_variable(4, np.float32)

    with C.layers.default_options(pad=True):
        conv1 = Conv((10,1), (4,), init=C.glorot_uniform(scale=init_scale))(input)
        pool1 = MaxPool((5,1), (2,1))(conv1)
        conv2 = Conv((10,1), (8,), init=C.glorot_uniform(scale=init_scale))(pool1)
        pool2 = MaxPool((5,1), (2,1))(conv2)
        conv3 = Conv((10,1), (16,), init=C.glorot_uniform(scale=init_scale))(pool2)
        pool3 = MaxPool((5,1), (2,1))(conv3)
        den1 = Dense(2048, activation=C.sigmoid)(pool3)
        den2 = Dense(2048, activation=C.sigmoid)(den1)
        z = Dense(4, activation=None)(den2)

    ce = C.cross_entropy_with_softmax(z, label)
    pe = C.classification_error(z, label)

    max_epoch = 60
    epoch_size = 40
    minibatch_size = 10

    lr_per_minibatch = learning_rate_schedule(0.0005, UnitType.minibatch)
    trainer = C.Trainer(z, ce, pe, sgd(z.parameters, lr=lr_per_minibatch))
    progress_printer = C.utils.ProgressPrinter(tag='Training')

    for i in range(max_epoch):
        for j in range(epoch_size):
            train_features, labels = generate_data(minibatch_size, train_data, train_label, 'train')
            trainer.train_minibatch({input : train_features, label : labels})
            progress_printer.update_with_trainer(trainer, with_metric=True)
        progress_printer.epoch_summary(with_metric=True)

    test_features, test_labels = generate_data(len(test_label), test_data, test_label, 'test')
    avg_error = trainer.test_minibatch({input : test_features, label : test_labels})
    print('Current accuracy on test: {}'.format(1. - avg_error))
    return avg_error

CNN()
