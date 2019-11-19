#!/usr/bin/python3

import argparse
import sys
import time

import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description="TP2 knn")
parser.add_argument("--visualize_data", type=int, metavar="data_id", help="Visualize image n and display corresponding class")
parser.add_argument("--compare", choices=["neighbors", "split", "distance"], help="Make one parameter vary to find the best outcome")

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