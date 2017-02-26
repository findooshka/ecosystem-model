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
    
    def accuracy(self, set):
        correct_predictions = 0
        for i in range(set.size()):
            prediction = self.predict(set.get_x(i))
            if (prediction * set.get_y(i) > 0):
                correct_predictions = correct_predictions + 1
        return correct_predictions / (set.size() * 1.0)

    def recall(self, set, positive_recall=True):
        correct_predictions = 0
        number_to_recall = 0
        for i in range(set.size()):
            prediction = self.predict(set.get_x(i))
            if (set.get_y(i) == 1 and positive_recall
                    or set.get_y(i) == -1 and not positive_recall):
                number_to_recall += 1
                if (prediction == 1 and positive_recall
                        or prediction == -1 and not positive_recall):
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
                    or prediction == -1 and not positive):
                total += 1
                if (set.get_y(i) == 1 and positive
                        or set.get_y(i) == -1 and not positive):
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
        
        def proceed(self, input_vector):
            if self.is_leaf:
                return self.label
            if input_vector[self.dimension] > self.separator:
                return self.right.proceed(input_vector)
            else:
                return self.left.proceed(input_vector)
    
    def __init__(self, input_dimension, leaf_threshold, metrics="Shannon"):
        self.input_dimension = input_dimension
        self.leaf_threshold = leaf_threshold
        if metrics != "Shannon" and metrics != "Gini":
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
    
    def predict(self, x):
        return self.root.proceed(x)

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
            
    
    def find_best_partition(self, current_set):
        min_entropy = 2
        current_min_dimension = -1
        current_min_separator = -1
        separator, left_side_count, right_side_count = (0, 0, 0)
        total_count = current_set.x.shape[0]
        for dimension in range(self.input_dimension):
            left_side_count, right_side_count = (0, 0)
            total_positives, total_negatives = self.count_results(current_set)
            for positive_count, negative_count, separator in self.move_partition(current_set, dimension):
                left_side_count = positive_count + negative_count
                right_side_count = current_set.x.shape[0] - left_side_count
                if self.metrics == "Shannon":
                    partition_entropy = (self.entropy(positive_count, negative_count) * left_side_count / total_count +
                                         self.entropy(total_positives - positive_count, total_negatives - negative_count) * left_side_count / total_count) / 2
                elif self.metrics == "Gini":
                    partition_entropy = (self.gini_metrics(positive_count, negative_count) * left_side_count / total_count +
                                         self.gini_metrics(total_positives - positive_count, total_negatives - negative_count) * right_side_count / total_count) / 2
                if partition_entropy < min_entropy:
                    min_entropy = partition_entropy
                    current_min_dimension = dimension
                    current_min_separator = separator
        return current_min_dimension, current_min_separator
                    
    def recurrent_train(self, current_set, current_vertex):
        if (self.metrics == "Shannon" and self.measure_set_entropy(current_set) <= self.leaf_threshold
               or self.metrics == "Gini" and self.measure_set_gini_metrics(current_set) <= self.leaf_threshold):
            current_vertex.set_label(1 if current_set.y.mean() > 0 else -1)
            return
        partition_parameters = self.find_best_partition(current_set)
        if partition_parameters[0] == -1:
            current_vertex.set_label(1 if current_set.y.mean() > 0 else -1)
            return
        left_vertex, right_vertex = current_vertex.create_descendants(partition_parameters[0],
                                                                      partition_parameters[1])
        subsets = self.partition(current_set, partition_parameters[0], partition_parameters[1])
        self.recurrent_train(subsets[0], left_vertex)
        self.recurrent_train(subsets[1], right_vertex)
    
    def train(self, training_set):
        self.root = DecisionTree.Vertex()
        self.recurrent_train(training_set, self.root)


class RandomForest(Classifier):
    def __init__(self, tree_count, tree_metrics="Shannon"):
        self.tree_count = tree_count
        self.tree_metrics = tree_metrics

    def predict(self, x):
        positive_trees_count = 0
        for i in range(self.tree_count):
            if self.trees[i].predict(x[self.trees_dimensions[i]]) == 1:
                positive_trees_count += 1
        return 1 if positive_trees_count > self.tree_count / 2 else -1

    def get_tree_input_dimesion(self, forest_input_dimension):
        return min(int(sqrt(forest_input_dimension)) + 2, forest_input_dimension)

    def train(self, labeled_set, verbose=False):
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
            self.trees.append(DecisionTree(self.tree_input_dimesion, 0.0, metrics=self.tree_metrics))
            self.trees_dimensions[i] = sorted(random.sample(range(self.input_dimension),
                                                     self.tree_input_dimesion))
            training_subset = LabeledSet(self.tree_input_dimesion)
            for j in range(labeled_set.size()):
                chosen_example_index = np.random.randint(labeled_set.size())
                training_subset.add_example(
                        labeled_set.get_x(chosen_example_index)[self.trees_dimensions[i]],
                        labeled_set.get_y(chosen_example_index)
                    )
            self.trees[i].train(training_subset)
            if verbose:
                print("trees trained:", (i + 1), "/", self.tree_count)