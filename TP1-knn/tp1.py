#!/usr/bin/python3

import argparse
import sys

import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description="TP1 knn")
parser.add_argument("--visualize", type=int, help="Visualize data")
parser.add_argument("--neighbors", type=int, default=10, help="Neighbor count")
parser.add_argument("--visualize_prediction", type=int, help="Visualize a prediction")
parser.add_argument("--best_k", action="store_true", help="Find best k by iterating from 2 to 15")

args = parser.parse_args()

# Get dataset
import sklearn.datasets as dsets

mnist = dsets.fetch_openml("mnist_784")

# Visualize data
def visualize(data, target):
    image = data.reshape((28, 28))
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    print(target)
    plt.show()

if args.visualize != None:
    visualize(images[args.visualize], mnist.target[args.visualize])
    sys.exit(0)

# knn
from sklearn.model_selection import train_test_split
import sklearn.neighbors as knn
import numpy as np

mnist_size = len(mnist.data)
subset_size = 5000
ids = np.random.randint(mnist_size, size=subset_size)
subset_data = mnist.data[ids]
subset_target = mnist.target[ids]

xtrain, xtest, ytrain, ytest = train_test_split(subset_data, subset_target, train_size=0.8)

def train_knn(neighbors, visualize):
    print("Training with n = %d" % (neighbors,))
    
    clf = knn.KNeighborsClassifier(neighbors)
    clf.fit(xtrain, ytrain)

    if visualize != None:
        visualize(xtest[visualize], clf.predict(xtest[visualize].reshape((1, -1))))

    print("Score for %d neighbors is %f" % (neighbors, clf.score(xtest, ytest)))
 
train_knn(args.neighbors, args.visualize_prediction)

# Test with multiple k
if args.best_k != None:
    pass