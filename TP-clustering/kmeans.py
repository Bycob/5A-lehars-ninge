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
parser.add_argument("--init", type=str, default="k-means++")
parser.add_argument("--max-clusters", type=int, default=20, help="Max amount of clusters in the data")
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
    best_db_score= 10000
    best_db = None
    best_sil_score = 0.0
    best_sil = None
    
    for i in range(2, args.max_clusters):
        km = KMeans(n_clusters = i, init = args.init)
        labels = km.fit_predict(data)
        db_score = metrics.davies_bouldin_score(data, labels)
        sil_score = metrics.silhouette_score(data, labels, metric="euclidean", sample_size=None, random_state=None)
        
        if db_score < best_db_score:
            best_db_score = db_score
            best_db = km
        
        if sil_score > best_sil_score:
            best_sil_score = sil_score
            best_sil = km

    labels = best_sil.fit_predict(data)
    print("Best silhouette score for cluster= %d is %f" %(best_sil.n_clusters, best_sil_score))
    
    labels = best_db.fit_predict(data)
    print("Best davies bouldin score for cluster= %d is %f" %(best_db.n_clusters, best_db_score))
    
    args.cluster = best_db.n_clusters
else:
    km = KMeans(n_clusters = args.cluster, init = args.init)
    labels = km.fit_predict(data)

print("cluster= %d" %(args.cluster))
#print("rand score : %f" %(metrics.adjusted_rand_score(labels_true, labels)))
print("davies bouldin score : %f" %(metrics.davies_bouldin_score(data, labels)))
print("silhouette score : %f" %(metrics.silhouette_score(data, labels, metric="euclidean", sample_size=None, random_state=None)))
visualize(x, y, labels, filename = "kmeans.png")
