import numpy as np
import random
import time

# x1_train = []
# x2_train = []
# y_train = []

# x1_test = []
# x2_test = []
# y_test = []

m = 10000  # number of training data
n = 500  # number of test data
k = 5000  # number of iterations
alpha = 0.01  # learning rate


def cross_entropy(a, y):
    return -(y*np.log10(a+1e-15) + (1-y)*np.log10(1-a+1e-15))


def sigmoid(z):
    return 1/(1+np.exp(-z))


# for i in range(m):
#     x1_train.append(random.uniform(-10, 10))
#     x2_train.append(random.uniform(-10, 10))
#     if x1_train[-1] + x2_train[-1] > 0:
#         y_train.append(1)
#     else:
#         y_train.append(0)

# for i in range(n):
#     x1_test.append(random.uniform(-10, 10))
#     x2_test.append(random.uniform(-10, 10))
#     if x1_test[-1] + x2_test[-1] > 0:
#         y_test.append(1)
#     else:
#         y_test.append(0)

# x_train = []  # for matrix calculation
# for i in range(m):
#     x_train.append([x1_train[i], x2_train[i]])

# x_test = []
# for i in range(n):
#     x_test.append([x1_test[i], x2_test[i]])

# x_train = np.array(x_train)
# x_train = x_train.transpose()

# x_test = np.array(x_test)

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

total_training_acc = 0
total_test_acc = 0
total_training_time = 0
total_test_time = 0
for _ in range(10):
    w = [random.uniform(-1, 1), random.uniform(-1, 1)]
    w = np.array(w).reshape(2, 1)
    b = random.uniform(-5, 5)
    training_start_time = time.time()
    for i in range(k):
        z = np.dot(w.transpose(), x_train) + b
        a = sigmoid(z)
        dz = a - y_train
        db = np.sum(dz, axis=1)/m
        dw = np.dot(x_train, dz.transpose())
        w -= alpha*dw
        b -= alpha*db
        # if i % 50 == 0:
        #     cost = np.sum(cross_entropy(a, y_train), axis=1)/m
        #     print(f"[W, b] : [{w.reshape(1, 2)}, {b}]\tCost : {cost}")
    total_training_time += time.time()-training_start_time

    x_train = x_train.transpose()
    train_correct_cnt = 0
    for i in range(m):
        z = np.dot(w.transpose(), x_train[i].transpose())
        a = sigmoid(z)
        if a >= 0.5 and y_train[i] == 1:
            train_correct_cnt += 1
        elif a < 0.5 and y_train[i] == 0:
            train_correct_cnt += 1
    print(f"Accuracy with training set : {train_correct_cnt/m*100}%")
    total_training_acc += train_correct_cnt/m*100

    test_correct_cnt = 0
    test_start_time = time.time()
    for i in range(n):
        z = np.dot(w.transpose(), x_test[i].transpose()) + b
        a = sigmoid(z)
        if a >= 0.5 and y_test[i] == 1:
            test_correct_cnt += 1
        elif a < 0.5 and y_test[i] == 0:
            test_correct_cnt += 1
    total_test_time += time.time() - test_start_time
    print(f"Accuracy with test set : {test_correct_cnt/n*100}%")
    total_test_acc += test_correct_cnt/n*100
    x_train = x_train.transpose()
print("Training Acc Average :", total_training_acc/10)
print("Test Acc Average :", total_test_acc/10)
print("Consuming Average Time in Training :", total_training_time/10)
print("Consuming Average Time in Test :", total_test_time/10)
