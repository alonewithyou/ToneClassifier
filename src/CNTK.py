import numpy as np
import cntk as C
import os
import data_utils
from cntk.learner import learning_rate_schedule, UnitType
from cntk.layers import Dense
from cntk.layers import Convolution as Conv
from cntk.layers import MaxPooling as MaxPool
from time import time

Counter = {'train': 0, 'test': 0}

config = {
    'New_length': 100,
    'exclude_st': 5.,
    'threshold_engy': 50.,
    'threshold_f0': 50.,
    'init_scale': 0.97,
    'channels': 1,
    'new_dim': 100,
    'cut_end': 3,
    'learning_rate': 5e-4
}
abs_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(abs_path, '..', 'toneclassifier', 'train')
test_path = os.path.join(abs_path, '..', 'toneclassifier', 'test_new')
train_label, train_data = data_utils.prepare(train_path, config)
test_label, test_data = data_utils.prepare(test_path, config)

def generate_data(sample_size, data, labels, type):
    topick = list(range(Counter[type], Counter[type] + sample_size))
    Counter[type] = (Counter[type] + sample_size) % len(labels)
    Y = np.array([[labels[i] == j for j in range(4)] for i in topick], dtype=np.float32)
    X = np.array([[data[i]] for i in topick])
    X = X.reshape((sample_size, config['channels'],-1,1))
    return X, Y


def CNN():
    input = C.ops.input_variable((config['channels'], config['new_dim'], 1), np.float32)
    label = C.ops.input_variable(4, np.float32)

    with C.layers.default_options(pad=True):
        conv1 = Conv((10,1), (2,), init=C.glorot_uniform(scale=config['init_scale']))(input)
        pool1 = MaxPool((4,1), (2,1))(conv1)
        conv2 = Conv((10,1), (4,), init=C.glorot_uniform(scale=config['init_scale']))(pool1)
        pool2 = MaxPool((4,1), (2,1))(conv2)
        conv3 = Conv((10,1), (8,), init=C.glorot_uniform(scale=config['init_scale']))(pool2)
        pool3 = MaxPool((4,1), (2,1))(conv3)
        den1 = Dense(1 << 11, activation=C.sigmoid)(pool3)
        den2 = Dense(1 << 11, activation=C.sigmoid)(den1)
        z = Dense(4, activation=None)(den2)

    ce = C.cross_entropy_with_softmax(z, label)
    pe = C.classification_error(z, label)

    max_epoch = 60
    epoch_size = 40
    minibatch_size = 10

    lr_per_batch = learning_rate_schedule(config['learning_rate'], UnitType.minibatch)
    momentum_plan = C.momentum_schedule(0.9)

    trainer = C.Trainer(z, ce, pe, C.sgd(z.parameters, lr=lr_per_batch))
                        #C.adam_sgd(z.parameters, lr=lr_per_batch, momentum=momentum_plan))
    progress_printer = C.utils.ProgressPrinter(tag='Training 5e-4')
    tic = time()
    for i in range(max_epoch):
        for j in range(epoch_size):
            train_features, labels = generate_data(minibatch_size, train_data, train_label, 'train')
            trainer.train_minibatch({input: train_features, label: labels})
            progress_printer.update_with_trainer(trainer, with_metric=True)
        progress_printer.epoch_summary(with_metric=True)
    toc = time()
    print('Total training time = %f s' % (-tic + toc))
    test_features, test_labels = generate_data(len(test_label), test_data, test_label, 'test')
    avg_error = trainer.test_minibatch({input: test_features, label: test_labels})
    print('Final accuracy on test: {}'.format(1. - avg_error))
    return avg_error

CNN()
