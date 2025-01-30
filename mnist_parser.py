import numpy as np
import random
import pandas as pd
import mnist

MNIST_DIR = r"C:\Users\kmkus_4e9n0iq\Desktop\Coding\AI\MNIST_images"

def parsed_MNIST():
    '''
    Returns a dictionary containing training data, training labels, testing data, and testing output
    Data is X, labels are Y
    The format of the training/testing data is a numpy 2d array with 784 rows and 8000/2000 columns
        The columns represent n_x, the number of features, while the 8000/2000 represent m, the # of training examples
    
    The format of the training/testing output is a numpy 2d array with 8000/2000 rows and 10 columns
        The data is stored in one-hot format, meaning that output[m][x] = 1 and everything else = 0
            In this example, the mth sample has a label of x
        
        The ROWS correspond to one of m training/testing examples, whereas the COLUMNS refer to the mth training example in the input data
    '''
    
    training_x = []
    training_y = []
    testing_x = []
    testing_y = []

    for i in range(1):
        f = open(f'C:\\Users\\kmkus_4e9n0iq\\Desktop\\Coding\\AI\\mnist-dataset\\data{i}.txt', "rb")
        inp = f.read()
        one_hot_formatted = [1 if idx == i else 0 for idx in range(10)]
        for j in range(800):
            selection = list(map(int, inp[(784 * j):(784 * (j + 1))]))
            training_x.append(np.array(selection) / 255.0)
            training_y.append(one_hot_formatted)
        
        for j in range(200):
            selection = list(map(int, inp[(784 * j):(784 * (j + 1))]))
            testing_x.append(np.array(selection) / 255.0)
            testing_y.append(one_hot_formatted)    
        
    return {
        "training_x" : np.array(training_x).T, 
        "training_y" : np.array(training_y).T, 
        "testing_x" : np.array(testing_x).T, 
        "testing_y" : np.array(testing_y).T
    }