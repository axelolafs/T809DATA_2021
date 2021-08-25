from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    N = len(targets)
    estimate = np.zeros(len(classes))
    i = 0
    for class_ in classes:
        j = 0
        for target in targets:
            if (target == class_):
                j += 1
        estimate[i] = j / N
        i += 1
    
    return estimate


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    
    N = len(features)
    features_1 = []
    targets_1 = []

    features_2 = []
    targets_2 = []

    for i in range(N):
        if(features[i][split_feature_index] < theta):
            features_1.append(features[i])
            targets_1.append(targets[i])
        else:
            features_2.append(features[i])
            targets_2.append(targets[i])

    return (np.array(features_1), np.array(targets_1)), (np.array(features_2), np.array(targets_2))


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    squarelist = np.power(prior(targets, classes), 2)
    sum = squarelist.sum()

    return 0.5*(1 - sum)


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    
    n1 = len(t1)
    n2 = len(t2)

    return ((n1*g1) + (n2*g2)) / n


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (_, t1), (_, t2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(t1, t2, classes)

def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        thetas = exclusive_interval(features[:,i], num_tries)
        # iterate thresholds
        for theta in thetas:
            current_gini = total_gini_impurity(features, targets, classes, i, theta)
            if(current_gini < best_gini):
                best_gini = current_gini
                best_dim = i
                best_theta = theta

    return best_gini, best_dim, best_theta

def exclusive_interval(values: np.ndarray, num_points: int) -> np.ndarray:
    '''
    Create 50 linearly spaced values between the minimum
    and maximum values in a given list. We also want to
    exclude the min and the max in the interval.
    '''
    min_value = values.min()
    max_value = values.max()

    interval = np.linspace(min_value, max_value, num_points+2)[1:-1]
    return interval

class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        return self.tree.fit(self.train_features,self.train_targets)

    def accuracy(self):
        ...

    def plot(self):
        ...

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        ...

    def guess(self):
        ...

    def confusion_matrix(self):
        ...


features, targets, classes = load_iris()
print(brute_best_split(features, targets, classes, 30))