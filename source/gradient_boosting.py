from basic_classes import Classifier, LabeledSet, Regression
from trees import DecisionTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt
import random

class GradientBoost(Regression):
    class ConstantClassifier(Classifier):
        def __init__(self):
            pass
        
        def train(self, training_set):
            self.result = training_set.y.mean()
            
        def predict(self, x):
            return self.result
    
    def __init__(self,
                 BaseClassifier):
        self.BaseClassifier = BaseClassifier
        self.activation_threshold = 0.0
    
    def set_activation_threshold(self, value, ):
        self.activation_threshold = value
    
    def regression_predict(self, x):
        return self.coefficients.dot([self.classifiers[i].predict(x) for i in range(len(self.classifiers))]) > self.activation_threshold
    
    def limited_prediction(self, x):  # utilisée pendant l'apprentissage
        return self.coefficients[:len(self.classifiers) - 1].dot([self.classifiers[i].predict(x) for i in range(len(self.classifiers) - 1)])
    
    def get_pseudo_residuals(self, loss_function_derivative, training_set):
        result = LabeledSet(training_set.x.shape[1])
        for example_index in range(training_set.size()):
            result.add_example(
                training_set.get_x(example_index),
                (-loss_function_derivative(self.limited_prediction(training_set.get_x(example_index)),
                                          training_set.get_y(example_index)))
            )
        return result
            
    def coefficient_gradient_descent(self,
                                     training_set,
                                     coefficient_index,
                                     loss_function_derivative,
                                     learning_rate,
                                     gradient_step_learning_rate,
                                     learn_threshold,
                                     max_descent_iterations):
        difference = learn_threshold + 0.000001
        classifier_prediction = 0.0
        previous_model_prediction = 0.0
        inital_coefficient = self.coefficients[coefficient_index]
        iteration = 0
        while abs(difference) > learn_threshold and max_descent_iterations > iteration:
            self.coefficients[coefficient_index] += difference
            difference = 0.0
            for example_index in range(training_set.size()):
                previous_model_prediction = self.limited_prediction(training_set.get_x(example_index))
                classifier_prediction = self.classifiers[-1].predict(training_set.get_x(example_index))
                difference -= (learning_rate
                               * classifier_prediction
                               * loss_function_derivative(classifier_prediction * self.coefficients[coefficient_index]
                                                          + previous_model_prediction,
                                                          training_set.get_y(example_index)))
            iteration += 1
        self.coefficients[coefficient_index] = (inital_coefficient
                                                + gradient_step_learning_rate
                                                * (self.coefficients[coefficient_index] - inital_coefficient))
    
    def train(self,
              training_set,
              learning_rate,
              learn_threshold,
              gradient_step_learning_rate,
              classifier_train_kwargs,
              classifier_init_args,
              classifier_init_kwargs,
              loss_function_derivative,  # d(L(ŷ, y)) / d(ŷ)
              iteration_count,
              max_descent_iterations=100000,
              verbose=False):
        self.classifiers = [self.ConstantClassifier()]
        self.classifiers[0].train(training_set)
        self.coefficients = np.zeros(iteration_count)
        self.coefficients[0] = 1
        for iteration in range(1, iteration_count):
            if verbose:
                print("Started iteration " + str(iteration))
            self.classifiers.append(self.BaseClassifier(*classifier_init_args, **classifier_init_kwargs))
            current_training_set = self.get_pseudo_residuals(loss_function_derivative, training_set)
            self.classifiers[-1].train(current_training_set, **classifier_train_kwargs)
            if verbose:
                print("Basic classifier trained")
            self.coefficient_gradient_descent(training_set,
                                              iteration,
                                              loss_function_derivative,
                                              learning_rate,
                                              gradient_step_learning_rate,
                                              learn_threshold,
                                              max_descent_iterations)
        if verbose:
            print("Finished training")
                


class TreeBoost(Regression):
    def __init__(self):
        self.BaseClassifier = DecisionTree
        self.activation_threshold = 0.0
        
    def limited_prediction(self, x):  # utilisée pendant l'apprentissage
        leaves_indexes = [0]
        for i in range(1, len(self.classifiers) - 1):
            leaves_indexes.append(self.classifiers[i].predict(x, find_index=True)[1])
        return self.coefficients[leaves_indexes].sum()
    
    def get_pseudo_residuals(self, loss_function_derivative, training_set):
        result = LabeledSet(training_set.x.shape[1])
        for example_index in range(training_set.size()):
            result.add_example(
                training_set.get_x(example_index),
                (-loss_function_derivative(self.limited_prediction(training_set.get_x(example_index)),
                                          training_set.get_y(example_index)))
            )
        return result
    
    def regression_predict(self, x):
        leaves_indexes = [0]
        for i in range(1, len(self.classifiers)):
            leaves_indexes.append(self.classifiers[i].predict(x, find_index=True)[1])
        return self.coefficients[leaves_indexes].sum() > self.activation_threshold
    
    def coefficient_gradient_descent(self,
                                     training_set,
                                     coefficient_index,
                                     loss_function_derivative,
                                     learning_rate,
                                     gradient_step_learning_rate,
                                     learn_threshold,
                                     max_descent_iterations):
        difference = (learn_threshold + 0.1) * np.ones(self.coefficients.shape[0] - self.previous_model_leaves_count)
        #inital_coefficients = self.coefficients[self.previous_model_leaves_count:]
        classifier_prediction = 0.0
        previous_model_prediction = 0.0
        iteration = 0
        while difference.dot(difference) > learn_threshold and max_descent_iterations > iteration:
            for i in range(difference.shape[0]):
                self.coefficients[self.previous_model_leaves_count + i] += difference[i] * gradient_step_learning_rate
                difference[i] = 0
            for example_index in range(training_set.size()):
                previous_model_prediction = self.limited_prediction(training_set.get_x(example_index))
                classifier_prediction = self.classifiers[-1].predict(training_set.get_x(example_index), find_index=True)
                difference[classifier_prediction[1] - self.previous_model_leaves_count] -= (
                    learning_rate
                    * loss_function_derivative(self.coefficients[classifier_prediction[1]]
                                               + previous_model_prediction,
                                               training_set.get_y(example_index))
                )
                iteration += 1
                
    def get_pseudo_residuals(self, loss_function_derivative, training_set):
        result = LabeledSet(training_set.x.shape[1])
        for example_index in range(training_set.size()):
            result.add_example(
                training_set.get_x(example_index),
                (-loss_function_derivative(self.limited_prediction(training_set.get_x(example_index)),
                                          training_set.get_y(example_index)))
            )
        return result
            
    def train(self,
              training_set,
              learning_rate,
              learn_threshold,
              gradient_step_learning_rate,
              classifier_train_kwargs,
              classifier_init_args,
              classifier_init_kwargs,
              loss_function_derivative,  # d(L(ŷ, y)) / d(ŷ)
              iteration_count,
              max_descent_iterations=500000,
              verbose=False):
        self.classifiers = [None]
        self.coefficients = np.zeros(1)
        self.coefficients[0] = training_set.y.mean()
        self.previous_model_leaves_count = 1
        for iteration in range(1, iteration_count + 1):
            if verbose:
                print("Started iteration " + str(iteration))
            self.classifiers.append(DecisionTree(*classifier_init_args, **classifier_init_kwargs))
            current_training_set = self.get_pseudo_residuals(loss_function_derivative, training_set)
            self.classifiers[-1].train(current_training_set, **classifier_train_kwargs)
            new_leaves_count = self.classifiers[-1].root.set_leaf_index(self.previous_model_leaves_count) - self.previous_model_leaves_count
            self.coefficients = np.hstack((self.coefficients, np.zeros(new_leaves_count)))
            if verbose:
                print("Tree is trained, calculating coefficient...")
            self.coefficient_gradient_descent(training_set,
                                              iteration,
                                              loss_function_derivative,
                                              learning_rate,
                                              gradient_step_learning_rate,
                                              learn_threshold,
                                              max_descent_iterations)
            self.previous_model_leaves_count += new_leaves_count
            if verbose:
                print("")
        if verbose:
            print("Finished training")