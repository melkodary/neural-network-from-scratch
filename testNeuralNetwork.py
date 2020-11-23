import numpy as np
import NeuralNetwork as Nn
from data.data_utils import load_CIFAR10

cifar10_dir = '../data/cifar-10-batches-py'
X_train, y_train, X_test, y_test, meta = load_CIFAR10(cifar10_dir)
classes = meta['label_names']

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


num_training = 40000

X_valid = X_train[num_training:]
y_valid = y_train[num_training:]

print('Pre-Processing data...')
X_train = Nn.NeuralNetwork.pre_process_data(X_train)
print('1 ')
X_valid = Nn.NeuralNetwork.pre_process_data(X_valid)
print('2 ')
X_test = Nn.NeuralNetwork.pre_process_data(X_test)
print('Done!')

X_train1 = X_train[:num_training]
y_train1 = y_train[:num_training]

learning_rate = 1e-1
batch_size = 50
epochs = 100
reg = 1e-3
drop_out = 0.8
alpha = 0.7
lr_decay = 0.5

count = 10
# for i in xrange(count):
# print '(', i, '/', count, ')',
# learning_rate = 10**np.random.uniform(-3, -1)
# reg = 10**np.random.uniform(-3, -2)
# print 'lr:', learning_rate, ', r:', reg,

my_nn = Nn.NeuralNetwork(X_train.shape[1], 10, learning_rate, reg, drop_out, alpha, lr_decay)
my_nn.add_layer(512)
my_nn.add_layer(128)
my_nn.train(X_train1, y_train1, X_valid, y_valid, epochs, batch_size)

my_nn.predict(X_test, y_test)
