import numpy as np
import matplotlib.pyplot as plt
from help import remove_one
from sklearn.metrics import accuracy_score
from tools import plot_points


from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    return np.linalg.norm(x-y)


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)
    sortedDistances = np.argsort(distances)
    return sortedDistances[0:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    ass = []
    for c in range(len(classes)):
        counter=0
        for t in range(len(targets)):
            if targets[t] == classes[c]:
                counter += 1     
        ass.append(counter)
    glassass = ass.index(max(ass))
    return classes[glassass]

def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    neighbors = k_nearest(x, points, k)
    label = vote(point_targets[neighbors], classes)
    return label


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    labels = []
    for i in range(len(point_targets)):
        tmp_points = remove_one(points, i)
        tmp_targets = remove_one(point_targets, i)
        labels.append(knn(points[i], tmp_points, tmp_targets, classes, k))
    return np.array(labels)


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    pred = knn_predict(points, point_targets, classes, k)
    return accuracy_score(point_targets, pred)


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    n = len(classes)
    matrix = np.zeros((n, n))
    N = len(point_targets)
    guessVector = knn_predict(points, point_targets, classes, k)

    for k in range(N):
        j = point_targets[k]
        i = guessVector[k]
        matrix[i][j] += 1
    return matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    N = len(point_targets)
    k_set = np.arange(1, N-1)
    k_best = 0
    max_accuracy = 0
    for z in k_set:
        accuracy = knn_accuracy(points, point_targets, classes, z)
        if max_accuracy < accuracy:
            k_best = z
            max_accuracy = accuracy
    return k_best

def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    colors = ['yellow', 'purple', 'blue']
    prediction = knn_predict(points, point_targets, classes, k)
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]

        if point_targets[i] == prediction[i]:
            edgestring = 'green'
        else:
            edgestring = 'red' 
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors=edgestring,
            linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section
    ...


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    ...


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    ...


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    ...

d, t, classes = load_iris()
(d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
knn_plot_points(d, t, classes, 3)