import mxnet as mx
import numpy as np
import pandas as pd
import logging
from time import time 

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
'''
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data, name='fc1', num_hidden=512)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="tanh")
fc2  = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 512)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type="tanh")
fc3  = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=4)
softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')'''

data = mx.symbol.Variable('data')

conv1 = mx.symbol.Convolution(data, kernel = (1,10), num_filter = 2,pad=(0,4))
pool1 = mx.symbol.Pooling(data = conv1, pool_type = "max", kernel = (1, 4), stride = (1, 2))

conv2 = mx.symbol.Convolution(pool1, kernel = (1,10), num_filter = 4,pad=(0,4))
pool2 = mx.symbol.Pooling(data = conv2, pool_type = "max", kernel = (1, 4), stride = (1, 2))

conv3 = mx.symbol.Convolution(pool2, kernel = (1,10), num_filter = 8,pad=(0,4))
pool3 = mx.symbol.Pooling(data = conv3, pool_type = "max", kernel = (1, 4), stride = (1, 2))

flatten = mx.symbol.Flatten(data = pool3)
fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 2048)
act1 = mx.symbol.Activation(data = fc1, act_type = "tanh")
fc2 = mx.symbol.FullyConnected(data = act1, num_hidden = 2048)
act2 = mx.symbol.Activation(data = fc2, act_type = "tanh")
fc3 = mx.symbol.FullyConnected(data = act2, num_hidden = 4)
softmax = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')


batch_size = 10
dataframe = pd.read_csv("../toneclassifier/train/datanew.csv", header=None)
X_train = dataframe.values
X_train = np.reshape(X_train, (X_train.shape[0], 1, 1, -1))
dataframe = pd.read_csv("../toneclassifier/train/labelnew.csv", header=None)
Y_train = dataframe.values
Y_train = np.reshape(Y_train,(-1))
train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size = batch_size)

dataframe = pd.read_csv("../toneclassifier/test_new/datanew.csv", header=None)
X_test = dataframe.values
X_test = np.reshape(X_test, (X_test.shape[0], 1, 1, -1))
dataframe = pd.read_csv("../toneclassifier/test_new/labelnew.csv", header=None)
Y_test = dataframe.values
Y_test = np.reshape(Y_test,(-1))
test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size = batch_size)

n_epoch = 60
mod = mx.mod.Module(softmax,context=mx.gpu(0))
tic = time()
mod.fit(train_iter,
        optimizer_params={'learning_rate': 0.01, 'momentum': 0}, 	num_epoch=n_epoch,initializer=mx.initializer.Normal(sigma=0.05))
toc = time()
print 'Total training time = %f' % (toc - tic)

metric = mx.metric.create('acc')
print mod.score(test_iter, metric)


