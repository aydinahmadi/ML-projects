#AydinAhmadi

import matplotlib.pyplot as plt
import numpy as np

labels = np.genfromtxt("hw06_true_labels.csv", delimiter = ",")

predicted_probs = np.genfromtxt("hw06_predicted_probabilities.csv", delimiter = ",")

N = (labels == -1).sum()
P = (labels == 1).sum()

arr = np.zeros((500, 2))
arr[:, 0] = predicted_probs
arr[:, 1] = labels
arr = arr[arr[:, 0].argsort()]


def draw_roc_curve(true_labels, predicted_probabilities):
    N = (true_labels == -1).sum()
    P = (true_labels == 1).sum()
    arr = np.zeros((500, 2))
    arr[:, 0] = predicted_probabilities
    arr[:, 1] = true_labels
    arr = arr[arr[:, 0].argsort()]

    points = []
    tpr_list = []
    fpr_list = []
    tnr_list = []
    fnr_list = []

    thresholds = arr[:, 0]
    print(f"thresholds:  {thresholds}")
    data_size = arr.shape[0]
    for threshold in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(data_size):
            if (arr[i][0] >= threshold):

                if arr[i][1] == 1:
                    tp = tp + 1
                if arr[i][1] == -1:
                    fp = fp + 1

            elif (arr[i][0] < threshold):

                if arr[i][1] == -1:
                    tn += 1
                else:
                    fn += 1

        fnr_list.append(fn / N)
        tnr_list.append(tn / P)

        fpr_list.append(fp / N)
        tpr_list.append(tp / P)

    plt.figure()
    plt.plot(fpr_list, tpr_list, "o-.", markersize=2, label="ROC")
    plt.xlabel("FP Rate")
    plt.ylabel("TP Rate")

    plt.title("ROC")
    plt.show()
    auc_roc = 0
    tpr_list.reverse()
    fpr_list.reverse()
    for i in range(data_size - 1):
        auc_roc += (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i] + tpr_list[i + 1]) / 2
    print(f"The area under the ROC curve is{auc_roc}")


draw_roc_curve(labels, predicted_probs)


def draw_pr_curve(true_labels, predicted_probabilities):
    N = (true_labels == -1).sum()
    P = (true_labels == 1).sum()
    # print(predicted_probabilities)
    arr = np.zeros((500, 2))
    arr[:, 0] = predicted_probabilities
    arr[:, 1] = true_labels
    arr = arr[arr[:, 0].argsort()]

    points = []
    recall = []
    precision = []

    thresholds = arr[:, 0]
    data_size = arr.shape[0]
    for threshold in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(data_size):
            if (arr[i][0] >= threshold):

                if arr[i][1] == 1:
                    tp = tp + 1
                if arr[i][1] == -1:
                    fp = fp + 1

            elif (arr[i][0] < threshold):

                if arr[i][1] == -1:
                    tn += 1
                else:
                    fn += 1

        precision.append(tp / (tp + fp))
        recall.append(tp / P)

    plt.figure()
    plt.plot(recall, precision, "o-", markersize=2, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("PR")
    plt.show()

    auc_pr = 0
    recall.reverse()
    precision.reverse()
    for i in range(data_size - 1):
        auc_pr += (recall[i + 1] - recall[i]) * (precision[i] + precision[i + 1]) / 2
    print(f"The area under the PR curve is {auc_pr}")


draw_pr_curve(labels, predicted_probs)
