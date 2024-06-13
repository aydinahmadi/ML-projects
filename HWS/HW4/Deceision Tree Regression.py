import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))


data_set= np.genfromtxt('hw04_data_set.csv',delimiter=',')


N = data_set.shape[0] - 1
print("number of datapoints: ", N)


X_train = data_set[1:151, 0]
Y_train = data_set[1:151, 1]

X_test = data_set[151:, 0]
Y_test = data_set[151:, 1]


#print("X train: ", X_train.shape)
#print("Y train: ", X_train.shape)
#print("X test: ", X_test.shape)
#print("Y test: ", X_test.shape)

def rmse(y_truth, prediction):
    return np.sqrt(np.mean((y_truth - prediction) ** 2))


def regression_tree(P):
    node_indices = {}
    is_terminal = {}
    need_split = {}
    node_means = {}
    node_splits = {}

    node_indices[1] = np.array(range(len(X_train)))
    is_terminal[1] = False
    need_split[1] = True

    while True:

        split_nodes = [key for key, value in need_split.items() if value == True]
        if len(split_nodes) == 0:
            break


        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_mean = np.mean(Y_train[data_indices])


            if X_train[data_indices].size <= P:
                is_terminal[split_node] = True
                node_means[split_node] = node_mean

            else:
                is_terminal[split_node] = False
                unique_val = np.sort(np.unique(X_train[data_indices]))
                split_positions = 0.5 * (unique_val[1:len(unique_val)] + unique_val[0:(len(unique_val) - 1)])
                split_scores = np.repeat(0.0, len(split_positions))

                for s in range(len(split_positions)):
                    left_indices = data_indices[X_train[data_indices] < split_positions[s]]
                    right_indices = data_indices[X_train[data_indices] >= split_positions[s]]
                    total_err = 0
                    if len(left_indices) > 0:
                        total_err += np.sum((Y_train[left_indices] - np.mean(Y_train[left_indices])) ** 2)
                    if len(right_indices) > 0:
                        total_err += np.sum((Y_train[right_indices] - np.mean(Y_train[right_indices])) ** 2)
                    split_scores[s] = total_err / (len(left_indices) + len(right_indices))


                if len(unique_val) == 1:
                    is_terminal[split_node] = True
                    node_means[split_node] = node_mean
                    continue
                best_split = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_split


                left_indices = data_indices[(X_train[data_indices] < best_split)]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True


                right_indices = data_indices[(X_train[data_indices] >= best_split)]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
    return node_means, node_splits, is_terminal


node_means, node_splits, is_terminal = regression_tree(25)


data_interval = np.linspace(np.minimum(np.min(X_train), np.min(X_test)), np.maximum(np.max(X_train), np.max(X_test)), 1001)

#print(data_interval)

def prediction(x, node_splits, node_means, is_terminal):
    index = 1  # start from root
    while 1:
        if is_terminal[index] == True:
            return node_means[index]
        if x > node_splits[index]:
            index = index * 2 + 1  # right child
        else:
                index = index * 2  # left child


y_hat_test = [prediction(x, node_splits, node_means, is_terminal) for x in data_interval]
y_pred_test = np.array(y_hat_test)

y_hat_train = [prediction(x, node_splits, node_means, is_terminal) for x in data_interval]
y_pred_train = np.array(y_hat_test)

left_borders = np.arange(0, 6, 0.01)
right_borders = np.arange(0 + 0.01, 6 + 0.01, 0.01)


plt.figure(figsize = (10, 6))
plt.plot(X_train, Y_train, "b.", markersize = 10, label = "training")
plt.plot(X_test, Y_test, "r.", markersize = 10, label = "test")
plt.plot(data_interval, y_pred_train,'k-',label= 'Prediction')

plt.xlabel("Eruption Time [min]")
plt.ylabel("Waiting Time to Next Eruption [min]")
plt.title("Decision Tree Prediction where P=25")
plt.title("P = 25")
plt.legend(loc = "upper left")
plt.show()

y_pred_train = [ prediction(x, node_splits, node_means, is_terminal) for x in X_train]
print("RMSE for training set= ",rmse(Y_train, y_pred_train))

y_pred_test = [ prediction(x, node_splits, node_means, is_terminal) for x in X_test]
print("RMSE for test set= ",rmse(Y_test, y_pred_test))


arr1 = np.arange(5, 55, 5)

rmse_train = []
rmse_test = []

for p in arr1:
    node_means, node_splits, is_terminal = regression_tree(p)
    y_pred_train = [prediction(x, node_splits, node_means, is_terminal) for x in X_train]
    y_pred_test = [prediction(x, node_splits, node_means, is_terminal) for x in X_test]

    rmse_train.append(rmse(Y_train, y_pred_train))
    rmse_test.append(rmse(Y_test, y_pred_test))

plt.figure(figsize = (8, 5))
plt.plot(arr1, rmse_train, color='blue' ,marker="o")
plt.plot(arr1, rmse_test,color='red' ,marker="o")
plt.xlabel("Pre-pruning size (P) ")
plt.ylabel("RMSE")
plt.legend(["training", "test"])

plt.show()

