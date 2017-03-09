import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt
import random


class LabeledSet:  
    def __init__(self, input_dimension):
        self.input_dimension=input_dimension
        self.example_number = 0
    
    def add_example(self, vector, label):
        if (self.example_number == 0):
            self.x = np.array([vector])
            self.y = np.array([label])
        else:
            self.x = np.vstack((self.x,vector))
            self.y = np.vstack((self.y,label))
        
        self.example_number = self.example_number + 1
    
    def get_input_dimension(self):
        return self.input_dimension
    
    def size(self):
        return self.example_number
    
    def get_x(self, i):
        return self.x[i]
    
    def get_y(self, i):
        return(self.y[i])


class Classifier:
    def __init__(self, input_dimension):
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        raise NotImplementedError("Please Implement this method")

    def train(self, training_set):
        raise NotImplementedError("Please Implement this method")
    
    def TP_FP_TN_FN(self, set):
        FN, FP, TP, TN = 0, 0, 0, 0
        for i in range(set.size()):
            if (self.predict(set.get_x(i)) != 1 and set.get_y(i) == 1):
                FN += + 1
            elif (self.predict(set.get_x(i)) == 1 and set.get_y(i) == 1):
                TP += + 1
            elif (self.predict(set.get_x(i)) != 1 and set.get_y(i) != 1):
                TN += + 1
            else:
                FP += + 1
        return TP, FP, TN, FN
    
    def precision(self, set):
        TP, FP, TN, FN = self.TP_FP_TN_FN(set)
        if (TP + FP != 0):
            return TP / (TP + FP)
        return -1

    def recall(self, set, positive_recall=True):
        TP, FP, TN, FN = self.TP_FP_TN_FN(set)
        if (TP + FN != 0):
            return TP / (TP + FN)
        return -1
    
    def fall_out(self, set):
        TP, FP, TN, FN = self.TP_FP_TN_FN(set)
        return FP / (FP + TN)

    def accuracy(self, set):
        TP, FP, TN, FN = self.TP_FP_TN_FN(set)
        return (TP + TN) / set.size()
    
    def f_measure(self, set, beta=1.0):
        if beta < 0:
            raise ValueError("Invalid beta: " + str(beta))
        precision = self.precision(set)
        recall = self.recall(set)
        if recall == 0 and beta * precision == 0:
            return 0
        return ((1 + beta ** 2) * precision * recall) / (precision * (beta ** 2) + recall)

class Regression(Classifier):
    def regression_predict(self, x):
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        return 1 if self.regression_predict(x) > self.activation_threshold else -1
        
    def set_activation_threshold(self, value):
        self.activation_threshold = value
        
    def plot_roc(self, labeled_set, point_count=150):
        current_threshold = self.activation_threshold
        tpr = []
        fpr = []
        for i in range(point_count, -1 - point_count, -2):
            self.activation_threshold = i / point_count
            tpr.append(self.recall(labeled_set))
            fpr.append(self.fall_out(labeled_set))
        plt.scatter(fpr, tpr)
        plt.show()
        auc_roc = 0
        for i in range(1, len(tpr)):
            auc_roc += (min(tpr[i], tpr[i - 1]) * (fpr[i] - fpr[i - 1])
                        + (max(tpr[i], tpr[i - 1]) - min(tpr[i], tpr[i - 1])) * (fpr[i] - fpr[i - 1]) / 2)
        self.activation_threshold = current_threshold
        return auc_roc


class Perceptron(Classifier):
    def __init__(self):
        pass
        
    def set_iteration_count(self, iteration_count):
        self.iteration_count = iteration_count
        
    def get_iteration_count(self):
        return self.iteration_count
        
    def predict(self, x):
        return self.activation_function(self.theta.dot(self.kernel_function(x)))
    
    def _predict_no_kernel(self, x):
        return self.activation_function(self.theta.dot(x))
    
    # adjust value of theta according to value of input vector
    def vector_train(self, input_vector, expected_output):
        self.theta -= (input_vector
                       * self.loss_function_derivative(self._predict_no_kernel(input_vector), expected_output)
                       * self.learning_rate)
    
    def train_iteration(self, labeledSet):
        chosen_index = np.random.randint(labeledSet.x.shape[0])
        self.vector_train(self.kernel_function(labeledSet.x[chosen_index]), labeledSet.y[chosen_index])
    
    def train(self,
              labeled_set,
              learning_rate,
              loss_function_derivative,
              iteration_count=10000,
              kernel_function=lambda x: x,
              activation_function=lambda x: 1 if x > 0 else -1):
        self.iteration_count = iteration_count
        self.learning_rate = learning_rate
        self.data_dimension = kernel_function(labeled_set.x[0]).shape[0]
        self.theta = np.ones(self.data_dimension)
        chosen_index = 0
        self.kernel_function = kernel_function
        self.activation_function = activation_function
        self.loss_function_derivative = loss_function_derivative
        for i in range(self.iteration_count):
            self.train_iteration(labeled_set)


def logistic_function(x):
        return 1 / (1 + np.exp(-x))
    
def least_squares_derivative(prediction, y):
    return 2 * (prediction - y)

def logistic_function_maximizer(prediction, y):
    return (logistic_function(prediction) - y)