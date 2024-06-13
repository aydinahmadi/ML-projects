import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safelog(x):
    return(np.log(x + 1e-100))

data_set= np.genfromtxt('hw03_data_set.csv',delimiter=',')


N = data_set.shape[0] - 1
print("number of datapoints: ", N)


X_train = data_set[1:151, 0].reshape(150,1)
Y_train = data_set[1:151, 1].reshape(150,1)

X_test = data_set[151:, 0].reshape(122,1)
Y_test = data_set[151:, 1].reshape(122,1)


print("X train: ", X_train.shape)
print("Y train: ", X_train.shape)
print("X test: ", X_test.shape)
print("Y test: ", X_test.shape)

bin_width = 0.37
origin = 1.5

left_borders = np.arange(origin, max(X_train), bin_width)
right_borders = np.arange(origin + bin_width, max(X_train) + bin_width, bin_width)
data_interval = np.arange(origin, 5.2, 0.001)
print("left_borders: ", left_borders)
print("right_borders: ",right_borders)


def regrerssogram():
    p_hat_regressogram = np.asanyarray(
        [np.sum(((left_borders[b] < X_train) & (X_train <= right_borders[b])) * Y_train) / \
         np.sum((left_borders[b] < X_train) & (X_train <= right_borders[b])) \
         for b in range(len(left_borders))])

    plt.figure()
    plt.plot(X_train, Y_train, "b.", markersize=10, label="Training")
    plt.plot(X_test, Y_test, "r.", markersize=10, label="Test")

    for b in range(len(left_borders)):
        plt.plot([left_borders[b], right_borders[b]], [p_hat_regressogram[b], p_hat_regressogram[b]], "k-")

    for b in range(len(left_borders) - 1):
        plt.plot([right_borders[b], right_borders[b]], [p_hat_regressogram[b], p_hat_regressogram[b + 1]], "k-")

    plt.xlabel("Eruption Time (min)")
    plt.ylabel("Waiting Time to Next Eruption (min)")
    plt.title("Regressogram")
    plt.legend()
    plt.show()
    return p_hat_regressogram


predicted_y_reg = regrerssogram()
print("predicted_y_reg: ", predicted_y_reg)
# RMSE Error

total_err = 0
for i in range(0, len(X_test)):
        loss = (Y_test[i] - predicted_y_reg[int((X_test[i] - origin) / bin_width)]) ** 2
        total_err += loss
rmse = np.sqrt(total_err / len(X_test))
print("\n")
print("Regressogram => RMSE is", float(rmse), " when h is", bin_width)
print("\n")


def mean_smoother():
    p_hat_mean_smoother = np.asanyarray(
        [np.sum((((x - 0.5 * bin_width) < X_train) & (X_train <= (x + 0.5 * bin_width))) * Y_train) \
         / np.sum(((x - 0.5 * bin_width) < X_train) & (X_train <= (x + 0.5 * bin_width))) \
         for x in data_interval])

    plt.figure()
    plt.plot(X_train, Y_train, "b.", markersize=10, label="Training")
    plt.plot(X_test, Y_test, "r.", markersize=10, label="Test")
    plt.plot(data_interval, p_hat_mean_smoother, "k-")

    plt.xlabel("Eruption Time (min)")
    plt.ylabel("Waiting Time to Next Eruption (min)")
    plt.title("Mean Smoother")
    plt.xlim([1.2, 5.5])
    plt.legend()
    plt.show()
    return p_hat_mean_smoother


predicted_y_mean_smoother = mean_smoother()
print("predicted_y_mean_smoother:", predicted_y_mean_smoother)


def kernal_smoother():
    p_hat_kernal = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * \
                                      np.exp(-0.5 * (x - X_train) ** 2 / bin_width ** 2) * Y_train) / np.sum(
        1.0 / np.sqrt(2 * math.pi) * \
        np.exp(-0.5 * (x - X_train) ** 2 / bin_width ** 2)) for x in data_interval])

    plt.figure()
    plt.plot(X_train, Y_train, "b.", markersize=10, label="Training")
    plt.plot(X_test, Y_test, "r.", markersize=10, label="Test")
    plt.plot(data_interval, p_hat_kernal, "k-")

    plt.xlabel("Eruption Time (min)")
    plt.ylabel("Waiting Time to Next Eruption (min)")
    plt.title("Kernel Smoother")
    plt.xlim([1.2, 5.5])
    plt.legend()
    plt.show()
    return p_hat_kernal


predicted_y_kernal_smoother = kernal_smoother()
print("predicted_y_kernal_smoother", predicted_y_kernal_smoother)


def kernal_smoother():
    p_hat_kernal = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * \
                                      np.exp(-0.5 * (x - X_train) ** 2 / bin_width ** 2) * Y_train) / np.sum(
        1.0 / np.sqrt(2 * math.pi) * \
        np.exp(-0.5 * (x - X_train) ** 2 / bin_width ** 2)) for x in data_interval])

    plt.figure()
    plt.plot(X_train, Y_train, "b.", markersize=10, label="Training")
    plt.plot(X_test, Y_test, "r.", markersize=10, label="Test")
    plt.plot(data_interval, p_hat_kernal, "k-")

    plt.xlabel("Eruption Time (min)")
    plt.ylabel("Waiting Time to Next Eruption (min)")
    plt.title("Kernel Smoother")
    plt.xlim([1.2, 5.5])
    plt.legend()
    plt.show()
    return p_hat_kernal


predicted_y_kernal_smoother = kernal_smoother()
#print(predicted_y_kernal_smoother)


def RMSE(x_test, y_test, y_hat):
    total_err = 0
    for i in range(0, len(x_test)):
            index = np.where(data_interval > x_test[i])[0][1]
            loss = (y_test[i] - y_hat[index])**2
            total_err += loss
    rmse = np.sqrt(total_err / len(x_test))
    return rmse

print("\n")
print("Mean Smoother  => RMSE is", float(RMSE(X_test, Y_test, predicted_y_mean_smoother)), " when h is", bin_width)
print("Kernel Smoother => RMSE is", float(RMSE(X_test, Y_test, predicted_y_kernal_smoother)), " when h is", bin_width)
