#!/usr/bin/python3

import argparse
import sys

import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description="TP1 knn")
parser.add_argument("--visualize_data", type=int, metavar="data_id", help="Visualize image n and display corresponding class")
parser.add_argument("--unique_train", type=int, metavar="k", help="Run an unique knn train with the specified k")
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

if args.visualize_data != None:
    visualize(images[args.visualize], mnist.target[args.visualize])
    sys.exit(0)

# knn
from sklearn.model_selection import train_test_split
import sklearn.neighbors as knn
import numpy as np

# Create Dataset
mnist_size = len(mnist.data)
subset_size = 5000
ids = np.random.randint(mnist_size, size=subset_size)
subset_data = mnist.data[ids]
subset_target = mnist.target[ids]

xtrain, xtest, ytrain, ytest = train_test_split(subset_data, subset_target, train_size=0.8)

def train_knn(neighbors):
    print("Training with k = %d" % (neighbors,))
    
    clf = knn.KNeighborsClassifier(neighbors)
    clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)

    print("Score for %d neighbors is %f" % (neighbors, score))
    return clf, score

if args.unique_train != None:
    knn0 = train_knn(args.unique_train)[0]
    visualize(xtest[0], knn0.predict(xtest[0].reshape((1, -1))))

# Test with multiple k
best_classifier = None
best_score = 0

if args.best_k != None:
    for k in range(2, 15):
        knn_k, score = train_knn(k)
        
        if score > best_score:
            best_score = score
            best_classifier = knn_k
            
print("Best classifier is k = %d with score: %f" % (best_classifier.n_neighbors, best_score))