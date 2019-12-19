#!/usr/bin/python3

# Run pip install --user hdbscan to be able to use hdbscan

import argparse
import matplotlib.pyplot as plt
import numpy as np

from datasets import *
import hdbscan

parser = argparse.ArgumentParser(description="TP clustering")
parser.add_argument("file", type=str, help="File to visualize")

args = parser.parse_args()

# Load dataset
x, y, _ = load_dset(args.file)
data = np.array([x, y]).transpose()

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
labels = clusterer.fit_predict(data)

visualize(x, y, labels, filename = "hdbscan.png")