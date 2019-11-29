from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.datasets as dsets
import sklearn.svm as svm
import sklearn.metrics as metrics
import sklearn.neighbors as knn
import sklearn.neural_network as n_network
import numpy as np
import time

# Load mnist
mnist = dsets.fetch_openml("mnist_784")

# Create dataset
mnist_size = len(mnist.data)
subset_size = 20000
ids = np.random.randint(mnist_size, size=subset_size)
subset_data = mnist.data[ids]
subset_target = mnist.target[ids]

xtrain, xtest, ytrain, ytest = train_test_split(subset_data, subset_target, train_size=0.8)

# Create the models
svc = svm.SVC(C=1.0, kernel="poly", degree = 3, gamma='auto')
mlpc = n_network.MLPClassifier(hidden_layer_sizes=tuple([50] * 20), activation="relu", solver="adam"
                               ,alpha=1e-4)
clf = knn.KNeighborsClassifier(3, p = 2)

print(["svm", "ann", "knn"])
models = [svc, mlpc, clf]
l_score = []
l_tt = []
l_st = []
l_recall = []
l_loss = []

for model in models:
    t = time.time()
    model.fit(xtrain, ytrain)
    training_t = time.time()
    l_score.append(model.score(xtest, ytest))
    scoring_t = time.time()
    ypred = model.predict(xtest)
    l_recall.append(metrics.recall_score(ytest, ypred, labels=None, pos_label=1, sample_weight=None, average='micro'))
    l_loss.append(metrics.zero_one_loss(ytest, ypred, normalize=True, sample_weight=None))
    matrix = metrics.confusion_matrix(ytest, ypred)
    print(matrix)
    l_tt.append(training_t - t)
    l_st.append(scoring_t - training_t)

print(l_score, l_tt, l_st, l_recall, l_loss, sep="\n")