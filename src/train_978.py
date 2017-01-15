import os
import random
import numpy as np
import cntk as C
from cntk.learner import learning_rate_schedule, UnitType
from cntk.layers import Dense
from cntk.layers import Convolution as Conv
from cntk.layers import MaxPooling as MaxPool
import scipy.interpolate as itp
import warnings

warnings.filterwarnings('ignore')

random.seed(23333)
New_length = 100
RESIZE = 100
channels = 1
en_norm_st = 700
f0_norm_st = 5000
exclude_st = 6.
silent_st = 0.05
threshold = 50.
Intervals = [10, 20]
init_scale = 0.97
Counter = {'train': 0, 'test': 0}


def pick_nonzero(feature):
    indice = np.where(feature > 0)
    if indice[0][-1] - indice[0][0] + 1 == len(indice[0]):
        return feature[indice]
    else:
        pos = [i for i in range(indice[0][0], indice[0][-1] + 1)]
        return feature[pos]


def exclude_noise(feature):
    if exclude_st < 0:
        return feature
    feature = np.array(feature)
    mean = np.abs(feature).mean()
    tmp = []
    for i in range(feature.shape[0]):
        if feature[i] < -exclude_st:
            pass
            tmp.append(-exclude_st)
        elif feature[i] > exclude_st:
            pass
            tmp.append(exclude_st)
        else:
            tmp.append(feature[i])
    return np.array(tmp)


def pre_processf0(feature):
    feature = pick_nonzero(feature)
    tmp = []
    for j in range(feature.shape[0] - 1):
        tmp.append(feature[j+1] - feature[j])
    feature = exclude_noise(np.array(tmp))
    curx = np.linspace(0, 100, feature.shape[0])
    newx = np.linspace(0, 100, New_length)
    fc = itp.interp1d(curx, feature, kind='cubic')
    feature = fc(newx)
    return feature


def exclude_by_engy(en, feature):

    feature = feature[np.where(en > threshold)[0]]
    feature = feature[3:-3]
    #print(feature)
    return feature


def fill(seq):
    while len(seq) < RESIZE:
        seq.append(0.)
    return seq


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

    ret1 = []
    ret2 = []
    for i in range(len(A)):
        cur_sz = len(B[i]) >> 1
        tmp1 = np.array(B[i][0:cur_sz], dtype='float64')
        tmp2 = np.array(B[i][cur_sz:], dtype='float64')

        ret1.append(A[i])
        tmp1 = exclude_by_engy(tmp2, tmp1)
        cut1 = pre_processf0(tmp1)

        tmp = list(cut1)

        ret2.append(tmp)

    ret1 = [int(s) for s in ret1]
    X = [i for i in range(len(ret1))]
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
    #print(topick)
    X = np.array([[data[i]] for i in topick])
    X = X.reshape((sample_size, channels, -1, 1))
    return X, Y


def CNN():
    input = C.ops.input_variable((channels, RESIZE,1), np.float32)
    label = C.ops.input_variable(4, np.float32)

    with C.layers.default_options(pad=True):
        conv1 = Conv((10,1), (2,), init=C.glorot_uniform(scale=init_scale))(input)
        pool1 = MaxPool((4,1), (2,1))(conv1)
        conv2 = Conv((10,1), (4,), init=C.glorot_uniform(scale=init_scale))(pool1)
        pool2 = MaxPool((4,1), (2,1))(conv2)
        conv3 = Conv((10,1), (8,), init=C.glorot_uniform(scale=init_scale))(pool2)
        pool3 = MaxPool((4,1), (2,1))(conv3)
        den1 = Dense(1 << 11, activation=C.sigmoid)(pool3)
        den2 = Dense(1 << 11, activation=C.sigmoid)(den1)
        z = Dense(4, activation=None)(den2)

    ce = C.cross_entropy_with_softmax(z, label)
    pe = C.classification_error(z, label)

    max_epoch = 60
    epoch_size = 40
    minibatch_size = 10

    lr_per_batch = learning_rate_schedule(0.0005, UnitType.minibatch)
    momentum_plan = C.momentum_schedule(0.9)

    trainer = C.Trainer(z, ce, pe, C.sgd(z.parameters, lr=lr_per_batch))
                        #C.adam_sgd(z.parameters, lr=lr_per_batch, momentum=momentum_plan))
    progress_printer = C.utils.ProgressPrinter(tag='Training 5e-4')
    for i in range(max_epoch):
        for j in range(epoch_size):
            train_features, labels = generate_data(minibatch_size, train_data, train_label, 'train')
            trainer.train_minibatch({input: train_features, label: labels})
            progress_printer.update_with_trainer(trainer, with_metric=True)
        progress_printer.epoch_summary(with_metric=True)

    test_features, test_labels = generate_data(len(test_label), test_data, test_label, 'test')
    avg_error = trainer.test_minibatch({input: test_features, label: test_labels})
    print('Final accuracy on test: {}'.format(1. - avg_error))
    return avg_error

CNN()
