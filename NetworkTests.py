from matplotlib.pyplot import axes
import numpy as np
from NeuralNetwork import load_network, NeuralNetwork
from SpiralData import *
from sklearn.utils import shuffle

def show_spiral_data(X,y):
    """Plots a spiral data set"""
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors=['k'], cmap=plt.cm.Spectral)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.show()

def test_digit_classifier(test_images, network, total_images):
    """Plots digits, once digit window is closed, the network prediction is printed"""
    test_images = shuffle(test_images)
    for i in range(total_images):
        plt.imshow(test_images[i].reshape(28,28), cmap='gray')
        plt.show()
        network.predict(test_images[i])

def test_network_accuracy(inputs, labels):
    """Tests a networks accuracy"""
    network.evaluate_network(inputs)
    accuracy = network.calculate_accuracy(labels)
    loss = network.calculate_loss(labels)
    print(f"Total Inputs: {inputs.shape[0]} \nAccuracy: {accuracy:.2f} Loss: {loss:.2f}")

def plot_classifier(network, X,y):
    """Plots the decision boundary of a network trained on the spiral dataset"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    input = np.c_[xx.ravel(), yy.ravel()]
    Z = network.evaluate_network(input)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors=['k'], cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

#Load digit data set
with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

#Reshape data to fit network
X = training_images.reshape(50000,784)
y = training_labels.reshape(50000,10)
y = y.astype(int)
test_labels = test_labels.reshape(test_labels.shape[0], 10)
test_labels = test_labels.astype(int)
test_images = test_images.reshape(test_images.shape[0], 784)

#Create spiral data set
#X, y = spiral_data(100,3)


#Training Settings
digit_classifier_name = "DigitClassifier2"
learning_rate = 1e-1
batch_size = 64
epoch = 200
shape = [784,128,128,10]
reg = 0

#Train Network

network = NeuralNetwork(digit_classifier_name, shape, reg)
network.train(X, y, learning_rate, epoch, batch_size, 10)
network.save_network()

#Test Network

#network = load_network(digit_name, reg)

test_digit_classifier(test_images, network, 20)
test_network_accuracy(test_images, test_labels)
