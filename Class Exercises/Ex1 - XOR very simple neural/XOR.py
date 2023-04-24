# -*- coding: utf-8 -*-


#   A very simple neural network to do exclusive or.
import numpy as np

inputLayerSize, hiddenLayerSize, outputLayerSize = 3, 3, 1

# prepare the dataset
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
Y = np.array([[0], [1], [1], [0]])

"""Complete the following functions

1.   sigmoid activation function
2.   Derivative of sigmoid
"""


# activation function
def sigmoid(x):
    # Your code here
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid
def sigmoid_(x):
    # Your code here
    return (1 - sigmoid(x)) * sigmoid(x)


#
# Test your sigmoid and sigmoid_ implementation
#
print(sigmoid(-10) < 6e-4)
print(sigmoid(10) - 0.9999 < 6e-4)
print(sigmoid(0) == 0.5)
print(sigmoid_(0) == 0.25)

"""# Train the network (Forward + backword)"""

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
Y = np.array([[0], [1], [1], [0]])
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))
epochs = 2000
for i in range(epochs):
    # first layer
    L1 = X @ Wh
    print(L1.shape == (4, 3))
    # sigmoid first layer results
    H = sigmoid(L1)
    print(H.shape == (4, 3))
    # second layer
    L2 = H @ Wz
    print(L2.shape == (4, 1))
    # sigmoid second layer results
    Z = sigmoid(L2)
    print(Z.shape == (4, 1))
    E = Y - Z  # how much we missed (error)
    print(E.shape == (4, 1))
    # gradient Z
    dZ = E * sigmoid_(L2)
    print(dZ.shape == (4, 1))
    # gradient H
    # the dz * Wz.T is the Error from the layer that the result save in l2
    dH = dZ * Wz.T * sigmoid_(L1)
    print(dH.shape == (4, 3))
    # update output layer weights
    Wz += H.T @ dZ
    print(Wz.shape == (3, 1))
    # update hidden layer weights
    Wh += X.T @ dH
    print(Wh.shape == (3, 3))

print(Z[0] < 0.05)  # what have we learnt?
print(Z[1] > 0.95)  # what have we learnt?
print(Z[2] > 0.95)  # what have we learnt?
print(Z[3] < 0.05)  # what have we learnt?
