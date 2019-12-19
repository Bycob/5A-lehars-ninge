#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
from datasets import *

from sklearn.cluster import DBSCAN
import sklearn.metrics as metrics

parser = argparse.ArgumentParser(description="TP clustering")
parser.add_argument("--eps", type=str, default="0.5", help="eps param for dbscan (default 0.5)")
parser.add_argument("--min-samples", type=str, default="5", help="min_samples param for dbscan (default 5)")
parser.add_argument("--criteria", type=str, default="db", help="Criteria for selecting best parameters: db = davies bouldin, sil = silhouette score")
parser.add_argument("--remove", action="store_true", help="Remove unassigned points before evaluating the model")
parser.add_argument("file", type=str, help="File to visualize")

args = parser.parse_args()

print("eps=%s, min-samples=%s" % (args.eps, args.min_samples))

# Preprocess arguments
eps_best = args.eps == "best"
min_samples_best = args.min_samples == "best"

if eps_best:
    args.eps = 0
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

if eps_best or min_samples_best:
    best_db_score= 1000
    best_db = None
    best_sil_score = -1000
    best_sil = None
    
    if eps_best:
        vals = np.linspace(0.1, 2.0, 20)
    else:
        vals = range(1, 21)
    
    for a in vals:
        eps = float(args.eps)
        min_samples = int(args.min_samples)
        
        if eps_best:
            eps = a
        else:
            min_samples = int(a)
        
        dbs = DBSCAN(eps = eps, min_samples=min_samples)
        labels = dbs.fit_predict(data)
        # TODO display unassigned labels so that we see which one they are.
        rdata, rlabels = remove_unassigned_points(data, labels) if args.remove else (data, labels)
        
        try:
            db_score = metrics.davies_bouldin_score(rdata, rlabels)
            sil_score = metrics.silhouette_score(rdata, rlabels, metric="euclidean", sample_size=None, random_state=None)
        except:
            print("eps == %f: only 1 cluster" % eps)
            continue
        
        """
        Problème avec les métriques vues précédemment: elles ne gèrent pas les points dans la catégorie "bruit"
        (et donc ont tendance à choisir le clustering avec le moins de bruit possible même s'il est très mauvais.)
        """
        
        print("eps == %f, min_samples == %f: davies bouldin is %f, silhouette is %f" % (eps, min_samples, db_score, sil_score))
        if db_score < best_db_score:
            best_db_score = db_score
            best_db = dbs
        
        if sil_score > best_sil_score:
            best_sil_score = sil_score
            best_sil = dbs
            
    best = best_db if args.criteria == "db" else best_sil
            
    print("Best is eps == %f and min_samples == %f" % (best.eps, best.min_samples))
    labels = best.fit_predict(data)
else:
    args.eps = float(args.eps)
    args.min_samples = int(args.min_samples)
    
    dbs = DBSCAN(eps = args.eps, min_samples=args.min_samples)
    labels = dbs.fit_predict(data)

print("Cluster count = %d" % cluster_count(labels))
visualize(x, y, labels, filename = "./dbscan.png")