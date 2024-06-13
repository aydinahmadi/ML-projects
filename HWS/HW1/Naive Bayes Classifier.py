#student name: Aydin Ahmadi

import numpy as np
import pandas as pd


def safelog(x):
    return(np.log(x + 1e-100))

df_data = pd.read_csv("D:\koc\ml\hw01_data_points.csv", sep=",", header=None)
df_labels = pd.read_csv("D:\koc\ml\hw01_class_labels.csv", sep=",", header=None)

data = df_data.to_numpy()
labels = df_labels.to_numpy()

train_labels = labels[:300]
test_labels = labels[300:]

all_data = np.append(data, labels, axis = 1)
training = all_data[:300]
test = all_data[300:]

print("dataset for training shape: ", training.shape)
print("dataset for test shape: ", test.shape)

characs = ['A', 'C', 'G', 'T']
labels_ = [1, 2]
training_size = training.shape[0]
features_size = training.shape[1]


def probability_estimation(column_charac, trainigsize, training_, features_size):
    p = []
    for j in labels_:
        for i in range(features_size - 1):
            pcol = [x[i] for x in training_ if x[i] == column_charac and x[7] == j]
            p.append(len(pcol) / (trainigsize / 2))
    p = np.reshape(p, (2, 7))
    return p


def prior_estimation(dataset):
    datalabels = []
    for j in labels_:
        labelsdata = [x for x in dataset if x[7] == j]
        datalabels.append(len(labelsdata) / len(dataset))

    return datalabels


p_Acd = probability_estimation('A', training_size, training, features_size)
p_Ccd = probability_estimation('C', training_size, training, features_size)
p_Gcd = probability_estimation('G', training_size, training, features_size)
p_Tcd = probability_estimation('T', training_size, training, features_size)

print("p_Acd: \n", p_Acd)
print("\n p_Ccd: \n", p_Ccd)
print("\n p_Gcd: \n", p_Gcd)
print("\n p_Tcd: \n", p_Tcd)

priors = prior_estimation(training)

print("\n class priors", priors)


def calculate_predictions(dataset):
    true1 = 0
    true2 = 0
    false1 = 0
    false2 = 0
    predictions = []
    for x in dataset:
        prior_prob = 0
        probs1 = []
        probs2 = []
        p1_Acd = [p_Acd[0][x_] for x_ in range(len(x)) if x[x_] == 'A']
        p2_Acd = [p_Acd[1][x_] for x_ in range(len(x)) if x[x_] == 'A']

        if p1_Acd != []:
            probs1.append(p1_Acd)
        if p2_Acd != []:
            probs2.append(p2_Acd)

        p1_Ccd = [p_Ccd[0][x_] for x_ in range(len(x)) if x[x_] == 'C']
        p2_Ccd = [p_Ccd[1][x_] for x_ in range(len(x)) if x[x_] == 'C']

        if p1_Ccd != []:
            probs1.append(p1_Ccd)
        if p2_Ccd != []:
            probs2.append(p2_Ccd)

        p1_Gcd = [p_Gcd[0][x_] for x_ in range(len(x)) if x[x_] == 'G']
        p2_Gcd = [p_Gcd[1][x_] for x_ in range(len(x)) if x[x_] == 'G']

        if p1_Gcd != []:
            probs1.append(p1_Gcd)
        if p2_Gcd != []:
            probs2.append(p2_Gcd)

        p1_Tcd = [p_Tcd[0][x_] for x_ in range(len(x)) if x[x_] == 'T']
        p2_Tcd = [p_Tcd[1][x_] for x_ in range(len(x)) if x[x_] == 'T']

        if p1_Tcd != []:
            probs1.append(p1_Tcd)
        if p2_Tcd != []:
            probs2.append(p2_Tcd)

        prior_prob1 = priors[0]
        prior_prob2 = priors[1]

        conditional_prob1 = np.prod([np.prod(probabilities) for probabilities in probs1])
        conditional_prob2 = np.prod([np.prod(probabilities) for probabilities in probs2])
        predict1 = safelog(conditional_prob1) + safelog(prior_prob1)
        predict2 = safelog(conditional_prob2) + safelog(prior_prob2)
        predict = 0
        if predict1 > predict2:
            predict = 1
        else:
            predict = 2
        predictions.append(predict)
    return predictions


def calculate_confusion_matrix(predictions_, labels):
    confusion_matrix = pd.crosstab(predictions_, labels.T,
                                   rownames=["y_pred"],
                                   colnames=["y_truth"])
    return confusion_matrix

train_predictions = calculate_predictions(training)
test_predictions = calculate_predictions(test)


train_confusion_matrix = calculate_confusion_matrix(train_predictions, train_labels)
test_confusion_matrix = calculate_confusion_matrix(test_predictions, test_labels)

print("train confusion matix:")
print(train_confusion_matrix)
print("train confusion matix:")
print(test_confusion_matrix)

