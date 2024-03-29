#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt

from scipy.io import arff

def load_dset(filename):
    dataset = arff.loadarff(open(filename, "r"))
    data = dataset[0]
    
    x = [e[0] for e in data]
    y = [e[1] for e in data]
    c = [e[2] for e in data]
    
    return x, y, c

def visualize(x, y, color, filename=None, outliers=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    color_valid = True
    
    for c in color:
        try:
            _ = int(c)
        except:
            color_valid = False
            print("Class format not recognized.")
            break
    
    if color_valid:
        ax.scatter(x, y, c = color, cmap= "tab20")
    else:
        ax.scatter(x, y)
        
    if filename != None:
        plt.savefig(filename)
        
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering datasets")
    parser.add_argument("file", type=str, help="File to visualize")
    
    args = parser.parse_args()
    
    dataset = load_dset(args.file)
    visualize(*dataset)
    
    