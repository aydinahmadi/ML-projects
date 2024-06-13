import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from scipy.stats import multivariate_normal
import matplotlib.transforms as transforms

X = np.genfromtxt("hw05_data_set.csv", delimiter = ",")
print(X.shape)


centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter = ",")
print(centroids.shape)

K = 5

x1 = X[:, 0]
x2 = X[:, 1]

N1 = N5 = 275
N2 = N3 = N4 = 150

N = 1000



means = [[0.0, 5.5],
        [-5.5, 0.0],
        [0.0, 0.0],
        [5.5, 0.0],
        [0.0, -5.5]]
covariances = [[[+0.8, -0.6], [-0.6, +0.8]],
               [[+0.8, +0.6], [+0.6, +0.8]],
               [[+0.8, -0.6], [-0.6, +0.8]],
               [[+0.8, +0.6], [+0.6, +0.8]],
               [[+1.6, +0.0], [+0.0, +1.6]]]

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:, 0], X[:, 1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12,
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

memberships = None
plot_current_state(centroids, memberships, X)

def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = centroids
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k, :], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)

memberships = update_memberships(centroids, X)


priors = [(memberships == c).sum()/len(memberships) for c in range (0,K)]
print(priors)

print(memberships)

covs = []
matrix_empt = np.zeros([2, 2], dtype = float)


for k in range(K):
    for i in range(X[memberships == k].shape[0]):
        cov = np.matmul(((X[memberships == k])[i,:] - centroids[k,:])[:, None], ((X[memberships == k])[i,:] - centroids[k,:][None, :]))
        matrix_empt += cov
    covs.append(matrix_empt / X[memberships == k].shape[0])
    matrix_empt = [[0.0, 0.0], [0.0, 0.0]]



initial_covariances = np.asanyarray(covs)
print(initial_covariances)


def update_means(h_ik, X):
    return(np.vstack([np.matmul(h_ik[k], X)/np.sum(h_ik[k], axis = 0) for k in range(K)]))


def update_covs(h_ik, means, X):
    covs = []
    empty_mat = [[0.0, 0.0], [0.0, 0.0]]
    for k in range(K):
        for i in range(N):
            cov = np.matmul((X[i] - means[k])[:, None], (X[i] - means[k])[None, :])*h_ik[k][i]
            empty_mat += cov
        covs.append(empty_mat / np.sum(h_ik[k], axis = 0))
        empty_mat = [[0.0, 0.0], [0.0, 0.0]]
    return(covs)


def update_priors(h_ik):
    return(np.vstack([np.sum(h_ik[k], axis = 0)/N for k in range(K)]))


iterations = 100
for iteration in range(iterations):
    # E-step
    post_probs = []
    for k in range(K):
        posterior = multivariate_normal(centroids[k], covs[k]).pdf(X) * priors[k]
        post_probs.append(posterior)
    h_ik = np.vstack([post_probs[k] / np.sum(post_probs, axis=0) for k in range(K)])
    # M-Step
    centroids = update_means(h_ik, X)
    covs = update_covs(h_ik, centroids, X)
    priors = update_priors(h_ik)

post_probs = []
for k in range(K):
    posterior = multivariate_normal(centroids[k], covs[k]).pdf(X) * priors[k]
    post_probs.append(posterior)
h_ik = np.vstack([post_probs[k] / np.sum(post_probs, axis=0) for k in range(K)])

print('centroids')
print(centroids)

memberships=np.argmax(h_ik, axis = 0)

means = [[0.0, 5.5],
         [-5.5, 0.0],
         [0.0, 0.0],
         [5.5, 0.0],
         [0.0, -5.5]]
covariances = [[[+0.8, -0.6], [-0.6, +0.8]],
               [[+0.8, +0.6], [+0.6, +0.8]],
               [[+0.8, -0.6], [-0.6, +0.8]],
               [[+0.8, +0.6], [+0.6, +0.8]],
               [[+1.6, +0.0], [+0.0, +1.6]]]

x, y = np.mgrid[-8:+8:.05, -8:+8:.05]
pos = np.dstack((x, y))

cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#9400d3"])
plt.figure(figsize=(10, 10))

for c in range(K):
    original_classes = multivariate_normal(means[c], np.array(covariances[c]) * 2).pdf(pos)
    found_classes = multivariate_normal(centroids[c], covs[c] * 2).pdf(pos)
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
             color=cluster_colors[c])
    plt.contour(x, y, original_classes, levels=1, linestyles="dashed", colors="k")
    plt.contour(x, y, found_classes, levels=1, colors=cluster_colors[c])

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
