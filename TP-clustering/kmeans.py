#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import arff
from sklearn.cluster import KMeans
from datasets import *
import sklearn.metrics as metrics

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

if args.best_cluster:
    best_i_score= 1.0
    best_i = 0
    best_j_score = 0.0
    best_j = 0
    for i in range(2, 11):
        km = KMeans(n_clusters = i, init = "k-means++")
        labels = km.fit_predict(data)
        db_score = metrics.davies_bouldin_score(data, labels)
        sil_score = metrics.silhouette_score(data, labels, metric="euclidean", sample_size=None, random_state=None)
        if db_score < best_i_score:
            best_i_score = db_score
            best_i = i
        if sil_score > best_j_score:
            best_j_score = metrics.davies_bouldin_score(data, labels)
            best_j = i

    km = KMeans(n_clusters=best_j, init="k-means++")
    labels = km.fit_predict(data)
    print("best silhouette score for cluster= %d is %f" %(best_j, best_j_score))
    
    km = KMeans(n_clusters=best_i, init="k-means++")
    labels = km.fit_predict(data)
    print("best davies bouldin score for cluster= %d is %f" %(best_i, best_i_score))

    visualize(x, y, labels)
else:
    km = KMeans(n_clusters = args.cluster, init = "k-means++")
    labels = km.fit_predict(data)
print("cluster= %d" %(args.cluster))
#print("rand score : %f" %(metrics.adjusted_rand_score(labels_true, labels)))
print("davies bouldin score : %f" %(metrics.davies_bouldin_score(data, labels)))
print("silhouette score : %f" %(metrics.silhouette_score(data, labels, metric="euclidean", sample_size=None, random_state=None)))
visualize(x, y, labels)
