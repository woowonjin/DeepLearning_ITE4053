import random
import numpy as np
import time
# for training
x1_train = []
x2_train = []
y_train = []

# for test
x1_test = []
x2_test = []
y_test = []

# number of training data
m = 1000

# number of test data
n = 100

# number of iterates
iterates_num = 2000

# learning rates
learning_rates = 0.005

for i in range(m):
    x1_train.append(random.uniform(-10, 10))
    x2_train.append(random.uniform(-10, 10))
    if x1_train[-1] + x2_train[-1] > 0:
        y_train.append(1)
    else:
        y_train.append(0)
    # print(x1_train[-1], x2_train[-1], y_train[-1])

for i in range(n):
    x1_test.append(random.uniform(-10, 10))
    x2_test.append(random.uniform(-10, 10))
    if x1_test[-1] + x2_test[-1] > 0:
        y_test.append(1)
    else:
        y_test.append(0)
    # print(x1_test[-1], x2_test[-1], y_test[-1])


def sigmoid(x):
    return 1/(1+np.exp(-x))


def cross_entropy_loss(a, y):
    return -(y*np.log10(a) + (1-y)*np.log10(1-a))


w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
b = random.uniform(-5, 5)


# Element wise Version
for j in range(iterates_num):
    d_w1 = 0
    d_w2 = 0
    d_b = 0
    J = 0
    w = np.array([w1, w2])
    for i in range(m):
        x = np.array([x1_train[i], x2_train[i]])
        z = np.dot(w, x) + b
        a = sigmoid(z)
        J += cross_entropy_loss(a, y_train[i])
        d_z = a - y_train[i]
        d_w1 += x1_train[i]*d_z
        d_w2 += x2_train[i]*d_z
        d_b += d_z
    J /= m
    d_w1 /= m
    d_w2 /= m
    d_b /= m
    w1 -= learning_rates*d_w1
    w2 -= learning_rates*d_w2
    b -= learning_rates*d_b
    if(j % 10 == 0):
        print(f"[w1, w2, b, J] = [{w1}, {w2}, {b}, {J}]")
    if(j == iterates_num-1):
        print("J_train =", J)

J_test = 0
w_result = np.array([w1, w2])
for i in range(n):
    x_test = np.array([x1_test[i], x2_test[i]])
    z = np.dot(w_result, x_test) + b
    a = sigmoid(z)
    J_test += cross_entropy_loss(a, y_test[i])
print("J_test =", J_test/n)


correct_train_cnt = 0
for i in range(m):  # training data
    x = np.array([x1_train[i], x2_train[i]])
    z = np.dot(w_result, x)
    a = sigmoid(z)
    if(z > 0.5 and y_train[i] == 1):
        correct_train_cnt += 1
    elif(z <= 0.5 and y_train[i] == 0):
        correct_train_cnt += 1

correct_test_cnt = 0
for i in range(n):  # test data
    x = np.array([x1_test[i], x2_test[i]])
    z = np.dot(w_result, x)
    a = sigmoid(z)
    if(z > 0.5 and y_test[i] == 1):
        correct_test_cnt += 1
    elif(z <= 0.5 and y_test[i] == 0):
        correct_test_cnt += 1

print("Accuracy for the m trainig data is " +
      str(correct_train_cnt/m*100) + "%")
print("Accuracy for the n test data is " + str(correct_test_cnt/n*100) + "%")

# ##############################Checking Time Consuming######################################
# # Vectorized Version
# vectorized_version_start_time = time.time()
# w_vectorized = np.array([w1, w2])
# b_vectorized = b
# x_all = []
# y_vectorized = np.array(y_train)
# for i in range(m):
#     x_all.append([x1_train[i], x2_train[i]])
# x_vectorized = np.array(x_all)  # m x 2
# x_vectorized = x_vectorized.transpose()

# for i in range(iterates_num):
#     z_vectorized = np.dot(w_vectorized, x_vectorized) + b
#     a_vectorized = sigmoid(z_vectorized)
#     dz_vectorized = a_vectorized - y_vectorized
#     dw_vectorized = np.dot(x_vectorized, dz_vectorized)/m
#     db_vectorized = np.sum(dz_vectorized)/m
#     w_vectorized = w_vectorized - learning_rates * dw_vectorized
#     b_vectorized = b_vectorized - learning_rates * db_vectorized
# print("Vectorized-Version consuming time :",
#       time.time()-vectorized_version_start_time)

# element_wise_version_start_time = time.time()
# # Element wise Version
# for j in range(iterates_num):
#     d_w1 = 0
#     d_w2 = 0
#     d_b = 0
#     w = np.array([w1, w2])
#     for i in range(m):
#         x = np.array([x1_train[i], x2_train[i]])
#         z = np.dot(w, x) + b
#         a = sigmoid(z)
#         d_z = a - y_train[i]
#         d_w1 += x1_train[i]*d_z
#         d_w2 += x2_train[i]*d_z
#         d_b += d_z
#     d_w1 /= m
#     d_w2 /= m
#     d_b /= m
#     w1 -= learning_rates*d_w1
#     w2 -= learning_rates*d_w2
#     b -= learning_rates*d_b
# print("Element-Wise-Version consuming time :",
#       time.time()-element_wise_version_start_time)
