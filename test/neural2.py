# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 2017
Neural network example
"""

import numpy as np
import mnist_loader

from network2 import Network

np.random.seed(420)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
validation_data = list(validation_data)


net = Network([784, 30, 30, 10])

'''
 Let's try training our network for 30 complete epochs, using mini-batches of 10 training examples
 at a time, a learning rate η=0.1, and regularization parameter λ=5.0.
 As we train we'll monitor the classification accuracy on the validation_data.
 '''
print('Network training started...\n')
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)
net.save('784_30_30_10')
# net.load('784_30_30_10.npz')

print('score = {:.2f} %'.format(net.accuracy(test_data)/100))
