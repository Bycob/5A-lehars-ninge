#!/usr/bin/python3

import argparse

import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description="TP1 knn")
parser.add_argument("--visualize", type=int, help="Visualize data")

args = parser.parse_args()

# Get dataset
import sklearn.datasets as dsets

mnist = dsets.fetch_openml("mnist_784")

# Visualize data
if args.visualize != None:
    images = mnist.data.reshape((-1, 28, 28))
    plt.imshow(images[args.visualize], cmap=plt.cm.gray_r,interpolation="nearest")
    print(mnist.target[args.visualize])
    plt.show()
