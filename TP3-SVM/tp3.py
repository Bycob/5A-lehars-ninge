#!/usr/bin/python3

import argparse
import sys
import time

import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description="TP3 SVM")
parser.add_argument("--visualize_data", type=int, metavar="data_id"
                    , help="Visualize image n and display corresponding class")
#parser.add_argument("--compare", choices=["neighbors", "split", "distance"]
# , help="Make one parameter vary to find the best outcome")
parser.add_argument("--c_param", type=float,default=1.0, help="Penalty parameter C of the error term")
parser.add_argument("--kernel", choices=["linear", "poly", "sigmoid","rbf","precomputed"], default="poly", help="Specifies the kernel type to be used in the algorithm")
parser.add_argument("--degree", type=int, default= 3, help="Degree of the polynomial kernel function ('poly')")


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
from sklearn.metrics import confusion_matrix
import sklearn.svm as svm
import sklearn.metrics as metrics
import numpy as np


#Create dataset
mnist_size = len(mnist.data)
subset_data = mnist.data
subset_target = mnist.target

xtrain, xtest, ytrain, ytest = train_test_split(subset_data, subset_target, train_size=49000/mnist_size)
print("train size = %d, test size = %d" % (len(xtrain), len(xtest)))

def classifier(c_param,kernel, degree):
    print("Training with kernel= %s ,c_param= %s, degree= %s (relevant for 'poly')" % (kernel, c_param, degree))
    svc=svm.SVC(C=c_param, kernel=kernel, degree = degree, gamma='auto')

    t = time.time()
    svc.fit(xtrain, ytrain)
    training_t = time.time()
    score = svc.score(xtest, ytest)
    scoring_t = time.time()
    ypred = svc.predict(xtest)
    recall = metrics.recall_score(ytest, ypred, labels=None, pos_label=1, sample_weight=None, average='micro')
    zero_one = metrics.zero_one_loss(ytest, ypred, normalize=True, sample_weight=None)
    matrix = metrics.confusion_matrix(ytest, ypred)

    print("The precision is : %f" % (score))
    print("Training time %f ms, scoring time %f ms" % ((training_t - t) * 1000, (scoring_t - training_t) * 1000))
    print("The recall score is : %f" %(recall))
    print("The zero-one classification loss is : %f" %(zero_one))

    print("The confusion matrix is :")
    print(matrix)
    return [svc, score, training_t, scoring_t,recall,zero_one]

c_param=[0.1,0.3,0.5,0.7,0.9]
kernel=["poly","linear", "sigmoid","precomputed","rbf"]
degree=[3,5,7,10,20]
score_c, training_t_c, scoring_t_c,recall_c,zero_one_c = ([] for i in range(5))
score_k, training_t_k, scoring_t_k,recall_k,zero_one_k = ([] for i in range(5))

for i in c_param:
    svc= classifier(i,args.kernel,args.degree)
    score_c.append(svc[1])
    training_t_c.append(svc[2])
    scoring_t_c.append(svc[3])
    recall_c.append(svc[4])
    zero_one_c.append(svc[5])
    
for i in kernel:
    svc= classifier(args.c_param,i,args.degree)
    score_k.append(svc[1])
    training_t_k.append(svc[2])
    scoring_t_k.append(svc[3])
    recall_k.append(svc[4])
    zero_one_k.append(svc[5])

print("With different values of C")
print("list of precisions")
print(score_c)
print("list of training times")
print(training_t_c)
print("list of scoring times")
print(scoring_t_c)
print("list of recall values")
print(recall_c)
print("list of zero-one parameters")
print(zero_one_c)

print("With different types of kernel")
print("list of precisions")
print(score_k)
print("list of training times")
print(training_t_k)
print("list of scoring times")
print(scoring_t_k)
print("list of recall values")
print(recall_k)
print("list of zero-one parameters")
print(zero_one_k)





#visualize(xtest[0], svc[0].predict(xtest[0].reshape((1, -1))))





