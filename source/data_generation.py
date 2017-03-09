from sklearn.datasets import fetch_mldata
from basic_classes import LabeledSet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt
import random

def generate_data_set():
    data_set = LabeledSet(2)
    sample = np.random.multivariate_normal((2, -2), 1.5 * np.identity(2), 100)
    for i in range(sample.shape[0]):
        data_set.add_example(sample[i], 1)
    sample = np.random.multivariate_normal((2, 2), np.identity(2), 100)
    for i in range(sample.shape[0]):
        data_set.add_example(sample[i], 1)
    sample = np.random.multivariate_normal((-2, -2), np.identity(2), 100)
    for i in range(sample.shape[0]):
        data_set.add_example(sample[i], 1)
    sample = np.random.multivariate_normal((-1.5, 1.5), np.identity(2), 100)
    for i in range(sample.shape[0]):
        data_set.add_example(sample[i], -1)
    return data_set

def plot_frontiere(set, classifier, step=20):
    mmax=set.x.max(0)
    mmin=set.x.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    plt.contourf(x1grid, x2grid, res)#, colors=["red","cyan"], levels=[-1000,0,1000], linewidth=2)
    
def plot2DSet(data_set):
    positive_indexes = []
    negative_indexes = []
    for i in range(data_set.y.shape[0]):
        if data_set.y[i] > 0:
            positive_indexes.append(i)
        else:
            negative_indexes.append(i)
    plt.scatter(data_set.x[positive_indexes, 0], data_set.x[positive_indexes, 1], marker='x')
    plt.scatter(data_set.x[negative_indexes, 0], data_set.x[negative_indexes, 1], marker='x')

def load_cancer_data(name):
    data = fetch_mldata(name, data_home='.')
    unique = np.unique(data.target)
    for i in range(len(data.target)):
        if (data.target[i]==unique[0]):
            data.target[i]=1
        else:
            data.target[i]=-1
    train_data = LabeledSet(data)
    for i in range(data.data.shape[0] // 2):
        train_data.add_example(data.data[i], data.target[i])
    test_data = LabeledSet(data)
    for i in range(data.data.shape[0] // 2, data.data.shape[0]):
        test_data.add_example(data.data[i], data.target[i])
    return (train_data, test_data)

def create_XOR(nb_points,var):
    data_set = LabeledSet(2)
    sample = np.random.multivariate_normal((0, 0), 0.01 * np.identity(2), 25)
    for i in range(sample.shape[0]):
        data_set.add_example(sample[i], 1)
    sample = np.random.multivariate_normal((1, 1), 0.01 * np.identity(2), 25)
    for i in range(sample.shape[0]):
        data_set.add_example(sample[i], 1)
    sample = np.random.multivariate_normal((1, 0), 0.01 * np.identity(2), 25)
    for i in range(sample.shape[0]):
        data_set.add_example(sample[i], -1)
    sample = np.random.multivariate_normal((0, 1), 0.01 * np.identity(2), 25)
    for i in range(sample.shape[0]):
        data_set.add_example(sample[i], -1)
    return data_set