# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 2017
Neural network example
"""

import numpy as np
import mnist_loader

from network import Network

np.random.seed(420)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

nn = Network([784, 30, 10])
nn.SGD(training_data, 30, 10, 2.5, test_data=test_data)
# nn.toFile('model.npz')
# nn.fromFile('model5.npz')

print('score = {:.2f} %'.format(nn.evaluate(test_data)/100))



# from PIL import Image
# 
# def imageToInput(filename):
#     im = Image.open(filename)
#     pix = im.load()
#     pix_data = np.array([ [(1-(sum(pix[x,y])/765))] for y in range(28) for x in range(28)])
#     max_value = max(pix_data)
#     if not max_value == 0:
#         pix_data = [i/max_value for i in pix_data]
#     return pix_data
# 
# def inputToImage(input_data, filename):
#     mode = 'L'
#     im = Image.new(mode, (28, 28))
#     pix = im.load()
#     for y in range(28):
#         for x in range(28):
#             value = int(255*(1-input_data[28*y+x][0]))
#             pix[x,y] = (value,)
#     im.save(filename)
# 
# def saveImages(test_data):
#     for i in range(len(test_data)):
#         filename = 'data/{}_{}.png'.format(test_data[i][1], i)
#         inputToImage(test_data[i][0], filename)
#         if i % 100 == 0:
#             print('{} images done!'.format(i))
#     print('Done!')
# 
# pix_data = imageToInput('test.png')
# results = nn.feedforward(pix_data)
# print('You wrote: {}'.format(np.argmax(results)))
# for i in range(10):
#     print('{}: {:4.1f} %'.format(i, 100*results[i][0]))

# from matplotlib import pyplot as plt
# 
# def plot_decision_boundary(pred_func, X, y):
#     # Set min and max values and give it some padding
#     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#     h = 0.01
#     # Generate a grid of points with distance h between them
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     # Predict the function value for the whole gid
#     Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     # Plot the contour and training examples
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
#     plt.show()