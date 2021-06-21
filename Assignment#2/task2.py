import numpy as np
import random
import time


def cross_entropy(a, y):
    return -(y*np.log10(a+1e-15) + (1-y)*np.log10(1-a+1e-15))


def sigmoid(z):
    return 1/(1+np.exp(-z))


m = 10000  # number of training data
n = 500  # number of test data
k = 5000  # number of iterations
alpha = 0.01  # learning rate

x_train = np.load("x_train.npy")  # (2,m)
y_train = np.load("y_train.npy")  # (1, m)
x_test = np.load("x_test.npy")  # (n, 2)
y_test = np.load("y_test.npy")  # (1, n)

total_training_acc = 0
total_test_acc = 0
total_training_time = 0
total_test_time = 0
for _ in range(10):
    w1 = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]
                  ).reshape(1, 2)  # 1, 2
    b1 = random.uniform(-5, 5)  # R
    w2 = random.uniform(-1, 1)  # R (1, 1)
    b2 = random.uniform(-5, 5)  # R

    training_start_time = time.time()
    for i in range(k):
        z1 = np.dot(w1, x_train) + b1  # (1, m)
        a1 = sigmoid(z1)  # (1, m)
        z2 = np.dot(w2, a1) + b2  # (1, m)
        a2 = sigmoid(z2)  # (1, m)
        dz2 = a2 - y_train  # (1, m)
        dw2 = np.dot(dz2, a1.transpose())/m  # R
        db2 = np.sum(dz2, axis=1)/m  # R
        da1 = np.dot(w2, dz2)  # (1, m)
        dz1 = da1*(a1*(1-a1))  # (1,m)
        dw1 = np.dot(dz1, x_train.transpose())/m
        db1 = np.sum(dz1, axis=1)/m
        w2 -= alpha*dw2
        b2 -= alpha*db2
        w1 -= alpha*dw1
        b1 -= alpha*db1
        # if i % 50 == 0:
        #     cost = np.sum(cross_entropy(a2, y_train), axis=1)/m
        #     print(f"W1 : {w1}, b1 : {b1}")
        #     print(f"W2 : {w2}, b2 : {b2}")
        #     print(f"Cost : {cost}")
    total_training_time += time.time()-training_start_time

    # test training set and test set
    train_correct_cnt = 0
    x_train = x_train.transpose()
    for i in range(m):
        z1 = np.dot(w1, x_train[i].transpose().reshape(2, 1)) + b1
        a1 = sigmoid(z1)  # (3, 1)
        z2 = np.dot(w2, a1) + b2  # R
        a2 = sigmoid(z2)
        if a2 >= 0.5 and y_train[i] == 1:
            train_correct_cnt += 1
        elif a2 < 0.5 and y_train[i] == 0:
            train_correct_cnt += 1
    print(f"Accuracy with training set : {train_correct_cnt/m*100}%")
    total_training_acc += train_correct_cnt/m*100

    test_correct_cnt = 0
    test_start_time = time.time()
    for i in range(n):
        z1 = np.dot(w1, x_test[i].transpose().reshape(2, 1)) + b1
        a1 = sigmoid(z1)  # (3, 1)
        z2 = np.dot(w2, a1) + b2  # R
        a2 = sigmoid(z2)
        if a2 >= 0.5 and y_test[i] == 1:
            test_correct_cnt += 1
        elif a2 < 0.5 and y_test[i] == 0:
            test_correct_cnt += 1
    total_test_time += time.time() - test_start_time
    print(f"Accuracy with test set : {test_correct_cnt/n*100}%")
    total_test_acc += test_correct_cnt/n*100
    x_train = x_train.transpose()


print("Training Acc Average :", total_training_acc/10)
print("Test Acc Average :", total_test_acc/10)
print("Consuming Average Time in Training :", total_training_time/10)
print("Consuming Average Time in Test :", total_test_time/10)
