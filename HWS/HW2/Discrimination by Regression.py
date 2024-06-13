import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



datapoints= np.genfromtxt('hw02_data_points.csv',delimiter=',')
labels= np.genfromtxt('hw02_class_labels.csv',delimiter=',')


data_set = np.hstack((datapoints, labels[:, None]))
print("dataset.shape: ", data_set.shape)

N = data_set.shape[0]
print("number of datapoints: ", N)

D = data_set.shape[1] - 1
print("number of features:", D)

X = data_set[:, :D]
print("X.shape: ", X.shape)
y_truth = data_set[:,D:(D+1)].astype(int)
print("y_truth.shape: ", y_truth.shape)


w = np.genfromtxt('hw02_W_initial.csv',delimiter=',')
w0 = np.genfromtxt('hw02_w0_initial.csv',delimiter=',')
w0 = w0.reshape((1,10))
print("w0.shape: ", w0.shape)
print("w.shape: ", w.shape)


def sigmoid(X, w, w0):
    return 1 / (1 + np.exp(-(np.matmul(X, w) + w0)))


def gradient_w(X, Y_truth, Y_predicted):
    return np.asarray([-np.matmul(Y_truth[:, c] - Y_predicted[:, c], X) for c in range(K)]).transpose()


def gradient_w0(Y_truth, Y_predicted):
    return np.sum(Y_truth - Y_predicted, axis = 0)


K = np.max(y_truth).astype(int)
print("number of classes: ", K)


Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth[:, 0] - 1] = 1
print("Y_truth.shape: ", Y_truth.shape)


train_X= X[:10000]
train_Y_truth = Y_truth[:10000]
train_y_truth = y_truth[:10000]
test_X = X[10000:]
test_Y_truth = Y_truth[10000:]
test_y_truth = y_truth[10000:]

print("train_X.shape: ", train_X.shape)
print("train_Y_truth: ", train_Y_truth.shape)
print("test_X.shape: ", test_X.shape)
print("test_Y_truth: ", test_Y_truth.shape)

eta = 0.00001
iteration_count = 1000


def update(iterations_, X, Y, y, w, w0):
    iteration = 1
    objective_values = []

    while iteration <= iterations_:
        Y_predicted = sigmoid(X, w, w0)
        objective_values = np.append(objective_values, np.sum(0.5 * (Y - Y_predicted) ** 2))

        w_new = w - eta * gradient_w(X, Y, Y_predicted)
        w0_new = w0 - eta * gradient_w0(Y, Y_predicted)

        w, w0 = w_new, w0_new
        iteration = iteration + 1

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, iteration), objective_values, "k-")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()

    print(Y_predicted.shape)
    y_predicted = np.argmax(Y_predicted, axis=1) + 1
    ypre = np.asarray(y_predicted)

    confusion_matrix = pd.crosstab(ypre, y.reshape((ypre.shape[0],)), rownames=['y_pred'], colnames=['y_truth'])
    print(confusion_matrix)

    return w, w0, Y_predicted


w_train, w0_train, train_predictions = update(iteration_count, train_X, train_Y_truth, train_y_truth ,w, w0)
print()
print("w_train", w_train)
print("w0_train", w0_train)
print()
w_test, w0_test, test_predictions = update(iteration_count, test_X, test_Y_truth, test_y_truth,w, w0)
print()
print("w_test", w_train)
print("w0_test", w0_train)
