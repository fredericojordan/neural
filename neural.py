# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 2017
Neural network example
"""

import numpy as np
# from matplotlib import pyplot as plt
import mnist_loader
# from PIL import Image

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                hits = self.evaluate(test_data)
                print("Epoch {}: {:.2f} %".format(j, 100*hits/n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    def toFile(self, filename):
        np.savez(filename, sizes=self.sizes, weights=self.weights, biases=self.biases)

    def fromFile(self, filename):
        model_file = np.load(filename)
        self.sizes = model_file['sizes']
        self.num_layers = len(self.sizes)
        self.weights = model_file['weights']
        self.biases = model_file['biases']

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

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

def loadTrainingData(filename):
    inputFile = open(filename, 'r')
    X = []
    y = []
    for line in inputFile.readlines():
        X.append([float(n) for n in line.split(',')[:-1]])
        y.append([int(line.split(',')[-1])])
    inputFile.close()
    return (np.array(X), np.array(y))

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

def saveImages(test_data):
    for i in range(len(test_data)):
        filename = 'data/{}_{}.png'.format(test_data[i][1], i)
        inputToImage(test_data[i][0], filename)
        if i % 100 == 0:
            print('{} images done!'.format(i))
    print('Done!')

def outputToResult(x):
    for i in range(len(x)):
        if x[i][0] == 1:
            return i

# X, y = loadTrainingData('training_data.txt')

# # Build a model with a 3-dimensional hidden layer
# model = build_model(X, y, 4, print_loss=True)
# # Plot the decision boundary
# plot_decision_boundary(lambda x: predict(model, x), X, y)

# nn = network.Network([2, 4, 1])
# training_data = list(zip(X, y))


np.random.seed(420)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
# pix_data = imageToInput('test.png')

nn = Network([784, 30, 10])
# nn.SGD(training_data, 30, 10, 2.5, test_data=test_data)
# nn.toFile('model.npz')
nn.fromFile('model5.npz')

print('score = {:.2f} %'.format(nn.evaluate(test_data)/100))
# results = nn.feedforward(pix_data)
# print('You wrote: {}'.format(np.argmax(results)))
# for i in range(10):
#     print('{}: {:4.1f} %'.format(i, 100*results[i][0]))
