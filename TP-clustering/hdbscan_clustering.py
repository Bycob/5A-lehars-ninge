#!/usr/bin/python3

# Run pip install --user hdbscan to be able to use hdbscan

import argparse
import matplotlib.pyplot as plt
import numpy as np

from datasets import *
import hdbscan
import sklearn.metrics as metrics

parser = argparse.ArgumentParser(description="TP clustering")
parser.add_argument("--min-cluster-size", type=str, default="5", help="Minimum cluster size")
parser.add_argument("--min-samples", type=str, default="1", help="")
parser.add_argument("--criteria", type=str, default="db", help="Criteria for selecting best parameters: db = davies bouldin, sil = silhouette score")
parser.add_argument("file", type=str, help="File to visualize")

args = parser.parse_args()

print("min-cluster-size=%s, min-samples=%s" % (args.min_cluster_size, args.min_samples))

min_cluster_size_best = args.min_cluster_size == "best"
min_samples_best = args.min_samples == "best"

if min_cluster_size_best:
    args.min_cluster_size = 0
if min_samples_best:
    args.min_samples = 0

# Load dataset
x, y, _ = load_dset(args.file)
data = np.array([x, y]).transpose()

def remove_unassigned_points(data, labels):
    rdata = []
    rlabels = []
    
    for i in range(len(data)):
        if labels[i] != -1:
            rlabels.append(labels[i])
            rdata.append(data[i])
    
    return rdata, rlabels

def cluster_count(labels):
    return max(labels) + 1

if min_cluster_size_best or min_samples_best:
    best_db_score= 1000
    best_db = None
    best_sil_score = -1000
    best_sil = None
    
    if min_cluster_size_best:
        vals = range(2, 50)
    else:
        vals = range(1, 20)
    
    for a in vals:
        min_cluster_size = int(args.min_cluster_size)
        min_samples = int(args.min_samples)
        
        if min_cluster_size_best:
            min_cluster_size = int(a)
        else:
            min_samples = int(a)
        
        hdbs = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
        labels = hdbs.fit_predict(data)
        # TODO display unassigned labels so that we see which one they are.
        rdata, rlabels = data, labels
        
        try:
            db_score = metrics.davies_bouldin_score(rdata, rlabels)
            sil_score = metrics.silhouette_score(rdata, rlabels, metric="euclidean", sample_size=None, random_state=None)
        except Exception as err:
            print("min_cluster_size == %f: only 1 cluster" % min_cluster_size)
            continue
        
        print("min_cluster_size == %f, min_samples == %f: davies bouldin is %f, silhouette is %f" % (min_cluster_size, min_samples, db_score, sil_score))
        if db_score < best_db_score:
            best_db_score = db_score
            best_db = hdbs
        
        if sil_score > best_sil_score:
            best_sil_score = sil_score
            best_sil = hdbs
            
    best = best_db if args.criteria == "db" else best_sil
            
    print("Best is min_cluster_size == %f and min_samples == %f" % (best.min_cluster_size, best.min_samples))
    labels = best.fit_predict(data)
else:
    args.min_cluster_size = int(args.min_cluster_size)
    args.min_samples = int(args.min_samples)
    
    hdbs = hdbscan.HDBSCAN(min_cluster_size = args.min_cluster_size, min_samples=args.min_samples, gen_min_span_tree=True)
    labels = hdbs.fit_predict(data)

print("Cluster count = %d" % cluster_count(labels))
visualize(x, y, labels, filename = "hdbscan.png")