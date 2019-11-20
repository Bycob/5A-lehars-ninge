#!/usr/bin/python3

import argparse
import sys
import time

import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description="TP2 knn")
parser.add_argument("--visualize_data", type=int, metavar="data_id"
                    , help="Visualize image n and display corresponding class")
#parser.add_argument("--compare", choices=["neighbors", "split", "distance"]
# , help="Make one parameter vary to find the best outcome")
parser.add_argument("--hidden_layer_sizes", type=list, default= [50], help="Tuple corresponding to the number of neural for each layer")
parser.add_argument("--activation", choices=["identity", "logistic", "tanh", "relu"], default="relu",
                    help="Activation function for the hidden layer")
parser.add_argument("--solver", choices=["ldfgs", "sgd", "adam"], default="adam", help="The solver for weight optimization")
parser.add_argument("--alpha", type=float,default=0.0001, help="L2 penalty (regularization term) parameter")

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

#MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.neural_network as n_network
import sklearn.metrics as metrics
import numpy as np


#Create dataset
mnist_size = len(mnist.data)
subset_data = mnist.data
subset_target = mnist.target

xtrain, xtest, ytrain, ytest = train_test_split(subset_data, subset_target, train_size=49000/mnist_size)
print("train size = %d, test size = %d" % (len(xtrain), len(xtest)))

def classifier(hidden_layer_sizes,activation, solver, alpha):
    print("Training with %d hidden layers, with the following neural number: %s" % (len(hidden_layer_sizes)
                                                                                    ,str(hidden_layer_sizes)))
    mlpc = n_network.MLPClassifier(hidden_layer_sizes=tuple(hidden_layer_sizes), activation=activation, solver=solver
                                   ,alpha=alpha)
    t = time.time()
    mlpc.fit(xtrain, ytrain)
    training_t = time.time()
    score = mlpc.score(xtest, ytest)
    scoring_t = time.time()

    print("The precision is %f with %d hidden layers, with the following neural number: %s" % (score, len(hidden_layer_sizes)
                                                                                    ,str(hidden_layer_sizes)))
    print("Training time %f ms, scoring time %f ms" % ((training_t - t) * 1000, (scoring_t - training_t) * 1000))
    print("The precision is :")
    #ypred = mlpc.predict(xtest)
    #print(metrics.precision_score(ytest, ypred, average='micro'))
    return mlpc, score


#create hidden_layer_sizes list
def layer(length):
    liste =[]
    for i in range(length):
        liste.append(50)
    return liste

# Run a normal test without making anything vary
print("With 1 layer of 50 neurals")
mlpc0 = classifier(args.hidden_layer_sizes, args.activation, args.solver, args.alpha)[0]
visualize(xtest[0], mlpc0.predict(xtest[0].reshape((1, -1))))

for i in [2, 10, 20, 50, 100]:
    print("With %d layer of 50 neurals" % (i,))
    mlpc = classifier(layer(i), args.activation, args.solver, args.alpha)[0]
    visualize(xtest[0], mlpc.predict(xtest[0].reshape((1, -1))))
