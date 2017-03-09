from basic_classes import Classifier, LabeledSet, Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt
import random

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
    
    def __init__(self, leaf_threshold, metrics="Shannon", vertex_possible_dimensions=-1):
        self.vertex_possible_dimensions = vertex_possible_dimensions
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
    
    def variance(self, data_set):  # pour la regression
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
            if data_set.y[sorted_indexes[index]] > 0:
                positive_count += 1
            else:
                negative_count += 1
            if np.array_equal(data_set.x[sorted_indexes[index], dimension], data_set.x[sorted_indexes[index + 1], dimension]):
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
            right_side_count = total_count - left_side_count
            if self.metrics == "Shannon":
                metrics = (self.entropy(positive_count, negative_count) * left_side_count / total_count +
                                     self.entropy(total_positives - positive_count, total_negatives - negative_count) * right_side_count / total_count) / 2
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
        dimensions = []
        if self.vertex_possible_dimensions == -1:
            dimensions = range(self.input_dimension)
        else:
            dimensions = random.sample(range(self.input_dimension), self.vertex_possible_dimensions)
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
        leaf.set_label(1 if current_set.y.mean() > 0 else -1)
                    
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
    
    def train(self, training_set, max_depth=-1, possible_partition_splits_count=15):
        self.input_dimension = training_set.x.shape[1]
        self.possible_partition_splits_count = possible_partition_splits_count
        self.root = DecisionTree.Vertex()
        self.max_depth = max_depth
        self.recurrent_train(training_set, self.root, 0)


class RandomForest(Classifier):
    def __init__(self, tree_metrics="Gini"):
        self.tree_metrics = tree_metrics

    def get_tree_count(self):
        return self.tree_count

    def get_tree_outputs(self, input_vector):
        return [self.trees[i].predict(x[self.trees_dimensions[i]]) for i in range(self.get_tree_count())]

    def predict(self, x):
        positive_trees_count = 0
        for i in range(self.tree_count):
            if self.trees[i].predict(x) == 1:#[self.trees_dimensions[i]]) == 1:
                positive_trees_count += 1
        return 1 if positive_trees_count > self.tree_count / 2 else -1

    def get_tree_input_dimesion(self, forest_input_dimension):
        return max(forest_input_dimension, 1)

    def train(self, labeled_set, tree_count=10, max_depth=-1, verbose=False):
        self.tree_count = tree_count
        if labeled_set.size() == 0:
            raise RuntimeError("Empty training set!")
        self.input_dimension = labeled_set.x.shape[1]
        self.tree_input_dimesion = self.get_tree_input_dimesion(self.input_dimension)
        self.trees = []
        chosen_example_index = 0
        for i in range(self.tree_count):
            self.trees.append(DecisionTree(0.0,
                                           metrics=self.tree_metrics,
                                           vertex_possible_dimensions=self.tree_input_dimesion))
            training_subset = LabeledSet(self.tree_input_dimesion)
            for j in range(labeled_set.size()):
                chosen_example_index = np.random.randint(labeled_set.size())
                training_subset.add_example(
                    labeled_set.get_x(chosen_example_index) ,
                    labeled_set.get_y(chosen_example_index)
                )
            self.trees[i].train(training_subset, max_depth=max_depth)
            if verbose:
                print("trees trained:", (i + 1), "/", self.tree_count)


