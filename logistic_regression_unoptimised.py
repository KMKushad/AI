import numpy as np
from math import log

#cumulative cost
J = 0

#weights matrix
W = np.array([0, 0])

#bias matrix
b = 0 

#derivatives of cost function with respect to weights and bias
dJ_dw = [0, 0]
dJ_db = 0

#learning rate
LEARNING_RATE = 0.01


X = np.array([
    [150, 70],
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])
y = np.array([0,1,1,0,0,1,1,0,1,0])
m = 10

#this is to make it easier to print out lists of matrices
def pretty_print(arr):
    for m in arr:
        print(f"{m}\n")

#sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#error in one training example
def loss_function(y_hat, y):
    return -(y * log(y_hat) + (1-y) * log(1 - y_hat))


def train():
    global W, b, J, dJ_dw, dJ_db
    for i in range(m):
        z = np.dot(W.T, X[i]) + b
        a = sigmoid(z)
        J += loss_function(a, y[i])
        dz = a - y[i]
        for j in range(len(W)):
            dJ_dw[j] += dz * X[i][j]
        dJ_db += dz
        J /= m
        dJ_db /= m
        for j in range(len(dJ_dw)):
            dJ_dw[j] /= m
            W[j] -= LEARNING_RATE * dJ_dw[j]
        
        b -= LEARNING_RATE * dJ_db

for i in range(10):
    train()
    print(f"Iteration: {i}, Cost: {J}")



