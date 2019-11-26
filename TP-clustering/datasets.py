#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt

from scipy.io import arff

def visualize(dataset):
    data = dataset[0]
    x = [e[0] for e in data]
    y = [e[1] for e in data]
    c = [e[2] for e in data]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(x, y, c = c)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering datasets")
    parser.add_argument("file", type=str, help="File to visualize")
    
    args = parser.parse_args()
    
    dataset = arff.loadarff(open(args.file, "r"))
    visualize(dataset)
    
    