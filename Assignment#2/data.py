import numpy as np
import random

m = 10000  # number of training data
n = 500  # number of test data

x1_train = []
x2_train = []
y_train = []

x1_test = []
x2_test = []
y_test = []

for i in range(m):
    x1_train.append(random.uniform(-10, 10))
    x2_train.append(random.uniform(-10, 10))
    if x1_train[-1] + x2_train[-1] > 0:
        y_train.append(1)
    else:
        y_train.append(0)

for i in range(n):
    x1_test.append(random.uniform(-10, 10))
    x2_test.append(random.uniform(-10, 10))
    if x1_test[-1] + x2_test[-1] > 0:
        y_test.append(1)
    else:
        y_test.append(0)

x_train = []  # for matrix calculation
for i in range(m):
    x_train.append([x1_train[i], x2_train[i]])

x_test = []
for i in range(n):
    x_test.append([x1_test[i], x2_test[i]])

x_train = np.array(x_train)
x_train = x_train.transpose()

x_test = np.array(x_test)

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)
