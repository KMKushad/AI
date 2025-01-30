import math
import numpy as np
from math import log
from mnist_parser import parsed_MNIST

np.random.seed(0)

#how many nodes in each layer
structure = np.array([784, 10, 10, 10])

#cumulative cost
J = 0

#weights matrix, contains structure[i] rows and structure[i + 1] columns for the the ith layer of the structure  
W = [np.random.rand(structure[i], structure[i+1]) for i in range(len(structure) - 1)]

#bias matrix
b = np.array([np.random.rand(structure[i + 1]) for i in range(len(structure) - 1)])

#derivatives of cost function with respect to weights and bias
dJ_dw = []
dJ_db = []

#learning rate
LEARNING_RATE = 0.01

#allows log(0) to be computed 
epsilon = 1e-7

#data processing
all_data = parsed_MNIST()
training_x, training_y, testing_x, testing_y = all_data["training_x"], all_data["training_y"], all_data["testing_x"], all_data["testing_y"]

#this is to make it easier to print out lists of matrices
def pretty_print(arr):
    for m in arr:
        print(f"{m}\n")

#sigmoid activation function
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

#cost (average error) of a training example
def cost_function(y_hat, y):
    #return np.sum(-(y * np.log(y_hat + epsilon) + (1-y) * np.log(1 - y_hat + epsilon)), axis=0) / len(y)
    return abs(np.sum(y_hat - y, axis = 1))

def train():
    m = training_x.shape[1]

    forward_prop_cache = [training_x]

    for i in range(len(structure) - 1): 
        Z = np.dot(W[i].T, forward_prop_cache[-1]) + b[i].reshape(10, 1)
        #print(Z.min())
        A = sigmoid(Z)
        #print(A.min())
        forward_prop_cache.append(A)
        #print(forward_prop_cache)
        #print(A.shape)
        #print(training_y.shape)
    
    J = cost_function(forward_prop_cache[-1], training_y)
    #print(J.shape)
    #print(J)

    dJ_dz = (forward_prop_cache[-1] - training_y)

    backward_prop_cache = [dJ_dz]

    for i in range(len(structure) - 1):
        dJ_dw = np.average(np.multiply(backward_prop_cache[-1], forward_prop_cache[-(i + 1)]), axis = 0)
        dJ_db = dJ_dz

    print(dJ_dw.shape)


    cache = 0

    return "sigma"

print(train())