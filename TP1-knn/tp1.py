#!/usr/bin/python3

import argparse
import sys
import time

import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description="TP1 knn")
parser.add_argument("--visualize_data", type=int, metavar="data_id", help="Visualize image n and display corresponding class")
parser.add_argument("--compare", choices=["neighbors", "split", "distance"], help="Make one parameter vary to find the best outcome")
parser.add_argument("--neighbors", type=int, metavar="k", default=10, help = "Number of neighbors")
parser.add_argument("--split", type=float, default=0.8, help="Percentage of data used for training")
parser.add_argument("--distance", metavar="p", type=float, default=2, help= "Exponent for Minkowski distance")
parser.add_argument("--multithread", action="store_true", help="Use multithreading")

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
    visualize(mnist.data[args.visualize_data], mnist.target[args.visualize_data])
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

n_jobs = -1 if args.multithread else 1

xtrain, xtest, ytrain, ytest = train_test_split(subset_data, subset_target, train_size=args.split)
print("train size = %d, test size = %d" % (len(xtrain), len(xtest)))

def train_knn(neighbors, distance):
    print("Training with k = %d, n_jobs = %d, minkowski distance with p = %f" % (neighbors, n_jobs, distance))
    
    clf = knn.KNeighborsClassifier(neighbors, n_jobs = n_jobs, p = distance)
    t = time.time()
    clf.fit(xtrain, ytrain)
    training_t = time.time()
    score = clf.score(xtest, ytest)
    scoring_t = time.time()

    print("Score for %d neighbors and p = %f is %f" % (neighbors, distance, score))
    print("Training time %f ms, scoring time %f ms" % ((training_t - t) * 1000, (scoring_t - training_t) * 1000))
    return clf, score

if args.compare == None:
    # Run a normal test without making anything vary
    knn0 = train_knn(args.neighbors, args.distance)[0]
    visualize(xtest[0], knn0.predict(xtest[0].reshape((1, -1))))
else:
    
    # Test with multiple k
    best_classifier = None
    best_score = 0
    best_split = args.split

    if args.compare == "neighbors":
        for k in range(2, 15):
            knn_k, score = train_knn(k, args.distance)
            
            if score > best_score:
                best_score = score
                best_classifier = knn_k
    elif args.compare == "distance":
        for p in [1, 2]:
            knn_k, score = train_knn(args.neighbors, p)
        
            if score > best_score:
                best_score = score
                best_classifier = knn_k
    elif args.compare == "split":
        for s in np.linspace(0.1, 0.9, 9):
            xtrain, xtest, ytrain, ytest = train_test_split(subset_data, subset_target, train_size=s)
            print("train size = %d, test size = %d" % (len(xtrain), len(xtest)))
            
            knn_k, score = train_knn(args.neighbors, args.distance)
            
            if score > best_score:
                best_score = score
                best_classifier = knn_k
                best_split = s
    
    print("Best classifier is k = %d, distance = %f, split = %f, with score: %f" % (best_classifier.n_neighbors, best_classifier.p, best_split, best_score))