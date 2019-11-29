import matplotlib.pyplot as plt

a = ["svm", "ann", "knn"]
b = [0.05800000000000005, 0.07899999999999996, 0.07099999999999995]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel("Model")
ax.set_ylabel("zero-one loss")
ax.bar(a, b)

plt.show()