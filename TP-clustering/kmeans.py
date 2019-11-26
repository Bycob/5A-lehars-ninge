#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import arff
from sklearn.cluster import KMeans
from datasets import *

parser = argparse.ArgumentParser(description="TP clustering")
parser.add_argument("--cluster", type=str, default="2", help="")
parser.add_argument("file", type=str, help="File to visualize")

args = parser.parse_args()

# Parse cluster count
args.best_cluster = args.cluster == "best"

if not args.best_cluster:
    args.cluster = int(args.cluster)

# Load dset and do kmeans
x, y, _ = load_dset(args.file)
data = np.array([x, y]).transpose()

km = KMeans(n_clusters = args.cluster, init = "k-means++")
labels = km.fit_predict(data)

visualize(x, y, labels)
