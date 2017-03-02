import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
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
    
    def accuracy(self, set):
        correct_predictions = 0
        for i in range(set.size()):
            prediction = self.predict(set.get_x(i))
            if (prediction == set.get_y(i)):
                correct_predictions = correct_predictions + 1
        return correct_predictions / (set.size() * 1.0)

    def recall(self, set, positive_recall=True):
        correct_predictions = 0
        number_to_recall = 0
        for i in range(set.size()):
            prediction = self.predict(set.get_x(i))
            if (set.get_y(i) == 1 and positive_recall
                    or set.get_y(i) == 0 and not positive_recall):
                number_to_recall += 1
                if (prediction == 1 and positive_recall
                        or prediction == 0 and not positive_recall):
                    correct_predictions += 1
        if number_to_recall == 0:
            return -1
        return correct_predictions / number_to_recall

    def precision(self, set, positive=True):
        correct_predictions = 0
        total = 0
        for i in range(set.size()):
            prediction = self.predict(set.get_x(i))
            if (prediction == 1 and positive
                    or prediction == 0 and not positive):
                total += 1
                if (set.get_y(i) == 1 and positive
                        or set.get_y(i) == 0 and not positive):
                    correct_predictions += 1
        if total == 0:
            return -1
        return correct_predictions / total
    
    def f_measure(self, set, positive=True, beta=1.0):
        if beta < 0:
            raise ValueError("Invalid beta: " + str(beta))
        precision = self.precision(set, positive)
        recall = self.recall(set, positive)
        if recall == 0 and beta * precision == 0:
            return 0
        return ((1 + beta ** 2) * precision * recall) / (precision * (beta ** 2) + recall)


class DecisionTree(Classifier):
    class Vertex:
        def __init__(self):
            self.is_leaf = True
        
        def create_descendants(self, dimension, separator):
            self.dimension = dimension
            self.separator = separator
            self.left = DecisionTree.Vertex()
            self.right = DecisionTree.Vertex()
            self.is_leaf = False
            return self.left, self.right
        
        def set_label(self, label):
            self.label = label
        
        def proceed(self, input_vector, find_index=False):
            if self.is_leaf:
                if find_index:
                    return (self.label, self.index)
                return self.label
            if input_vector[self.dimension] > self.separator:
                return self.right.proceed(input_vector, find_index)
            else:
                return self.left.proceed(input_vector, find_index)
        
        def copy(self):
            vertex = Vertex()
            if self.is_leaf:
                vertex.set_label(self.label)
            else:
                vertex.left = self.left.copy()
                vertex.right = self.right.copy()
                vertex.dimension = dimension
                vertex.separator = separator
        
        # fonction pour TreeBoost 
        def set_leaf_index(self, index):
            if self.is_leaf:
                self.index = index
                return index + 1
            index = self.left.set_leaf_index(index)
            index = self.right.set_leaf_index(index)
            return index
    
    def __init__(self, leaf_threshold, metrics="Shannon"):
        self.leaf_threshold = leaf_threshold
        if metrics != "Shannon" and metrics != "Gini" and metrics != "Variance":
            raise ValueError("Unrecognized metrics: " + metrics)
        self.metrics = metrics
    
    def count_results(self, data_set):
        positive_count = 0
        negative_count = 0
        for output in data_set.y:
            if output > 0:
                positive_count += 1
            else:
                negative_count += 1
        return positive_count, negative_count
    
    #### Metrics ####
    
    def gini_metrics(self, positive_count, negative_count):
        total_count = positive_count + negative_count
        return (1
                - (positive_count / total_count) ** 2
                - (negative_count / total_count) ** 2) 
    
    def measure_set_gini_metrics(self, data_set):
        positive_count, negative_count = self.count_results(data_set)
        return self.gini_metrics(positive_count, negative_count)
    
    def entropy(self, positive_count, negative_count):
        positive_probability = positive_count / (positive_count + negative_count)
        if positive_probability == 0 or positive_probability == 1:
            return 0
        return -(positive_probability * log(positive_probability, 2)
                + (1 - positive_probability) * log(1 - positive_probability, 2))
    
    def measure_set_entropy(self, data_set):
        positive_count, negative_count = self.count_results(data_set)
        return self.entropy(positive_count, negative_count)
    
    def variance(self, data_set):
        return data_set.y.var() * data_set.size()
        
    ##################
    
    def predict(self, x, find_index=False):
        return self.root.proceed(x, find_index)

    def partition(self, data_set, separator_dimension, separator):
        left_set = LabeledSet(self.input_dimension)
        right_set = LabeledSet(self.input_dimension)
        for i in range(data_set.x.shape[0]):
            if data_set.x[i, separator_dimension] <= separator:
                left_set.add_example(data_set.x[i], data_set.y[i])
            else:
                right_set.add_example(data_set.x[i], data_set.y[i])
        return left_set, right_set
    
    def move_partition(self, data_set, dimension):
        sorted_indexes = np.argsort(data_set.x[:, dimension])
        positive_count, negative_count = 0, 0
        for index in range(sorted_indexes.shape[0] - 1):
            if data_set.y[sorted_indexes[index]] == 1:
                positive_count += 1
            else:
                negative_count += 1
            if (data_set.x[sorted_indexes[index], dimension] == data_set.x[sorted_indexes[index + 1], dimension]):
                continue
            yield (positive_count,
                   negative_count,
                   (data_set.x[sorted_indexes[index], dimension] + data_set.x[sorted_indexes[index + 1], dimension]) / 2)
    
    def find_classification_best_axis_partition(self,
                                                current_set,
                                                dimension,
                                                current_min_dimension,
                                                current_min_separator,
                                                current_min_metrics):
        total_positives, total_negatives = self.count_results(current_set)
        total_count = current_set.x.shape[0]
        left_side_count, right_side_count = (0, 0)
        metrics = 0
        for positive_count, negative_count, separator in self.move_partition(current_set, dimension):
            left_side_count = positive_count + negative_count
            right_side_count = current_set.x.shape[0] - left_side_count
            if self.metrics == "Shannon":
                metrics = (self.entropy(positive_count, negative_count) * left_side_count / total_count +
                                     self.entropy(total_positives - positive_count, total_negatives - negative_count) * left_side_count / total_count) / 2
            elif self.metrics == "Gini":
                metrics = (self.gini_metrics(positive_count, negative_count) * left_side_count / total_count +
                                     self.gini_metrics(total_positives - positive_count, total_negatives - negative_count) * right_side_count / total_count) / 2
            if metrics < current_min_metrics:
                current_min_metrics = metrics
                current_min_dimension = dimension
                current_min_separator = separator
        return current_min_dimension, current_min_separator, current_min_metrics
    
    def find_regression_best_axis_partition(self,
                                            current_set,
                                            dimension,
                                            current_min_dimension,
                                            current_min_separator,
                                            current_min_metrics):
        metrics = 0
        for separator in np.linspace(current_set.x[:, dimension].min(),
                                     current_set.x[:, dimension].max(),
                                     self.possible_partition_splits_count + 1):
            left_set, right_set = self.partition(current_set, dimension, separator)
            if left_set.size() == 0 or right_set.size() == 0:
                continue
            if self.metrics == "Variance":
                metrics = self.variance(left_set) + self.variance(right_set)
            if metrics < current_min_metrics:
                current_min_metrics = metrics
                current_min_dimension = dimension
                current_min_separator = separator
        return current_min_dimension, current_min_separator, current_min_metrics
    
    def find_best_partition(self, current_set):
        current_min_metrics = 2
        if self.metrics == "Variance":
            current_min_metrics = self.variance(current_set)
        current_min_dimension = -1
        current_min_separator = -1
        for dimension in range(self.input_dimension):
            if self.metrics == "Shannon" or self.metrics == "Gini":
                (
                    current_min_dimension,
                    current_min_separator,
                    current_min_metrics
                ) = self.find_classification_best_axis_partition(current_set,
                                                            dimension,
                                                            current_min_dimension,
                                                            current_min_separator,
                                                            current_min_metrics)
            elif self.metrics == "Variance":
                (
                    current_min_dimension,
                    current_min_separator,
                    current_min_metrics
                ) = self.find_regression_best_axis_partition(current_set,
                                                        dimension,
                                                        current_min_dimension,
                                                        current_min_separator,
                                                        current_min_metrics)
        return current_min_dimension, current_min_separator
    
    def set_leaf_label(self, leaf, current_set):
        if self.metrics == "Variance":
            leaf.set_label(current_set.y.mean())
            return
        leaf.set_label(1 if current_set.y.mean() > 0 else 0)
                    
    def recurrent_train(self, current_set, current_vertex, current_depth):
        if (self.metrics == "Shannon" and self.measure_set_entropy(current_set) <= self.leaf_threshold
               or self.metrics == "Gini" and self.measure_set_gini_metrics(current_set) <= self.leaf_threshold):
            self.set_leaf_label(current_vertex, current_set)
            return
        if (self.metrics == "Variance" and self.variance(current_set) <= self.leaf_threshold):
            self.set_leaf_label(current_vertex, current_set)
            return
        partition_parameters = self.find_best_partition(current_set)
        if partition_parameters[0] == -1:
            self.set_leaf_label(current_vertex, current_set)
            return
        left_vertex, right_vertex = current_vertex.create_descendants(partition_parameters[0],
                                                                      partition_parameters[1])
        subsets = self.partition(current_set, partition_parameters[0], partition_parameters[1])
        if self.max_depth == -1 or current_depth < self.max_depth:
            self.recurrent_train(subsets[0], left_vertex, current_depth + 1)
            self.recurrent_train(subsets[1], right_vertex, current_depth + 1)
        else:
            self.set_leaf_label(left_vertex, subsets[0])
            self.set_leaf_label(right_vertex, subsets[1])
    
    def train(self, training_set, max_depth=-1, possible_partition_splits_count=10):
        self.input_dimension = training_set.x.shape[1]
        self.possible_partition_splits_count = possible_partition_splits_count
        self.root = DecisionTree.Vertex()
        self.max_depth = max_depth
        self.recurrent_train(training_set, self.root, 0)


class RandomForest(Classifier):
    def __init__(self, tree_count, tree_metrics="Gini"):
        self.tree_count = tree_count
        self.tree_metrics = tree_metrics

    def get_tree_count(self):
        return self.tree_count

    def get_tree_outputs(self, input_vector):
        return [self.trees[i].predict(x[self.trees_dimensions[i]]) for i in range(self.get_tree_count())]

    def predict(self, x):
        positive_trees_count = 0
        for i in range(self.tree_count):
            if self.trees[i].predict(x[self.trees_dimensions[i]]) == 1:
                positive_trees_count += 1
        return 1 if positive_trees_count > self.tree_count / 2 else -1

    def get_tree_input_dimesion(self, forest_input_dimension):
        return min(int(sqrt(forest_input_dimension)) + 2, forest_input_dimension)

    def train(self, labeled_set, max_depth=-1, verbose=False):
        if labeled_set.size() == 0:
            raise RuntimeError("Empty training set!")
        self.input_dimension = labeled_set.x.shape[1]
        self.tree_input_dimesion = self.get_tree_input_dimesion(self.input_dimension)
        self.trees = []
        self.trees_dimensions = np.zeros((
                                             self.tree_count,
                                             self.tree_input_dimesion,
                                         )).astype(int)
        chosen_example_index = 0
        for i in range(self.tree_count):
            self.trees.append(DecisionTree(0.0, metrics=self.tree_metrics))
            self.trees_dimensions[i] = sorted(random.sample(range(self.input_dimension),
                                                     self.tree_input_dimesion))
            training_subset = LabeledSet(self.tree_input_dimesion)
            for j in range(labeled_set.size()):
                chosen_example_index = np.random.randint(labeled_set.size())
                training_subset.add_example(
                        labeled_set.get_x(chosen_example_index)[self.trees_dimensions[i]],
                        labeled_set.get_y(chosen_example_index)
                    )
            self.trees[i].train(training_subset, max_depth=max_depth)
            if verbose:
                print("trees trained:", (i + 1), "/", self.tree_count)

    # fonctionnes pour TreeBoost
    def enumerate_leaves(self):
        leaf_index = 0
        for tree_index in range(len(self.trees)):
            leaf_index = self.trees[tree_index].root.set_leaf_index(leaf_index)
        return leaf_index
    
    def get_predictions_and_index(self, x):
        results = []
        for i in range(self.tree_count):
            results.append(self.trees[i].predict(x[self.trees_dimensions[i]], find_index=True))
        return np.array(results)


class GradientBoost(Classifier):
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
    
    def predict(self, x):
        return 1 if self.coefficients.dot([self.classifiers[i].predict(x) for i in range(len(self.classifiers))]) > 0.1 else -1
    
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
                                     learn_threshold):
        difference = learn_threshold + 0.1
        classifier_prediction = 0.0
        previous_model_prediction = 0.0
        while abs(difference) > learn_threshold:
            self.coefficients[coefficient_index] += difference
            difference = 0.0
            for example_index in range(training_set.size()):
                previous_model_prediction = self.limited_prediction(training_set.get_x(example_index))
                classifier_prediction = self.classifiers[-1].predict(training_set.get_x(example_index))
                difference -= (learning_rate
                               * self.classifiers[-1].predict(training_set.get_x(example_index))
                               * loss_function_derivative(classifier_prediction * self.coefficients[coefficient_index]
                                                          + previous_model_prediction,
                                                          training_set.get_y(example_index)))
    
    def train(self,
              training_set,
              learning_rate,
              learn_threshold,
              classifier_train_kwargs,
              classifier_init_args,
              classifier_init_kwargs,
              loss_function_derivative,  # d(L(ŷ, y)) / d(ŷ)
              iteration_count,
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
            self.coefficient_gradient_descent(training_set, iteration, loss_function_derivative, learning_rate, learn_threshold)
        if verbose:
            print("Finished training")
                