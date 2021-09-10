from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    features_of_class = []
    for i in range(len(targets)):
        if targets[i] == selected_class:
            features_of_class.append(features[i])
    return np.mean(features_of_class, axis=0)



def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    features_of_class = []
    for i in range(len(targets)):
        if targets[i] == selected_class:
            features_of_class.append(features[i])
    return np.cov(features_of_class, rowvar=False)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return multivariate_normal(mean=class_mean
    , cov=class_covar).pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    likelihoods = []
    for i in range(test_features.shape[0]):
        vec = []
        for j in range(len(classes)):
            vec.append(likelihood_of_class(test_features[i], means[j], covs[j]))
        likelihoods.append(vec)
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    return np.argmax(likelihoods, axis=1)


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    likelihoods = []
    count_class = np.bincount(train_targets)
    print(count_class)
    for i in range(test_features.shape[0]):
        vec = []
        for j in range(len(classes)):
            scaler = count_class[j]/len(train_targets)

            vec.append(likelihood_of_class(test_features[i], means[j], covs[j])* scaler)
        likelihoods.append(vec)
    return np.array(likelihoods)

def accuracy(test_targets, likelihoods):
    prediction = predict(likelihoods)

    return accuracy_score(test_targets, prediction)

def confusion_matrix(classes, test_targets, likelihoods):
    n = len(classes)
    matrix = np.zeros((n, n))
    N = len(test_targets)
    guessVector = predict(likelihoods)

    for k in range(N):
        i = test_targets[k]
        j = guessVector[k]
        matrix[i][j] += 1
    return matrix

def main():
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.6)
    mle = maximum_likelihood(train_features, train_targets, test_features, classes)
    map = maximum_aposteriori(train_features, train_targets, test_features, classes)
    print(mle)
    print(map)
    
    print(accuracy(test_targets, mle))
    print(accuracy(test_targets, map))

    print(confusion_matrix(classes, test_targets, mle))
    print(confusion_matrix(classes, test_targets, map))

if __name__ == '__main__':
    main()