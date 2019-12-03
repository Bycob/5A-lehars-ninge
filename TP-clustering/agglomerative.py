#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import arff
from sklearn.cluster import AgglomerativeClustering
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

# Load dset and do agglomerative clustering
x, y, _ = load_dset(args.file)
data = np.array([x, y]).transpose()

if args.best_cluster:
    best_db_score = 1.0
    best_db = 0
    best_sil_score = 0.0
    best_sil = 0
    for link in ['ward', 'complete', 'average', 'single']:
        for i in range(2, 20):
            ac = AgglomerativeClustering(n_clusters=i, linkage=link)
            labels = ac.fit_predict(data)
            db_score = metrics.davies_bouldin_score(data, labels)
            sil_score = metrics.silhouette_score(data, labels, metric="euclidean", sample_size=None, random_state=None)

            if db_score < best_db_score:
                best_db_score = db_score
                best_db = i

            if sil_score > best_sil_score:
                best_sil_score = sil_score
                best_sil = i

        ac = AgglomerativeClustering(n_clusters=best_sil, linkage=link)
        labels = ac.fit_predict(data)
        #The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
        print("Best silhouette score for cluster= %d is %f (linkage= %s)" % (best_sil, best_sil_score, link))

        ac = AgglomerativeClustering(n_clusters=best_db, linkage=link)
        labels = ac.fit_predict(data)
        #The minimum score is zero, with lower values indicating better clustering.
        print("Best davies bouldin score for cluster= %d is %f (linkage= %s)" % (best_db, best_db_score, link))

        args.cluster = best_db
else:
    for link in ['ward', 'complete', 'average', 'single']:
        ac = AgglomerativeClustering(n_clusters=args.cluster, linkage=link)
        labels = ac.fit_predict(data)

        print("cluster= %d and linkage= %s" % (args.cluster,link))
        # print("rand score : %f" %(metrics.adjusted_rand_score(labels_true, labels)))
        print("davies bouldin score : %f" % (metrics.davies_bouldin_score(data, labels)))
        print("silhouette score : %f" % (
            metrics.silhouette_score(data, labels, metric="euclidean", sample_size=None, random_state=None)))
        visualize(x, y, labels)


