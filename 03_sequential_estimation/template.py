from numpy.core.numeric import identity
from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''

    X = []
    Ik = identity(k)
    cov = np.power(var, 2) * Ik
    for i in range(n):
        X.append(np.random.multivariate_normal(mean, cov))

    return np.array(X)

def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + (1/n)*(x-mu)


def _plot_sequence_estimate():
    data = gen_data(100, 3, np.array([0, 0, 0]), 1.3)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i], i+1))

    # print(estimates)
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.xlabel('Iteration')
    plt.ylabel('Estimated mean')
    plt.show()
    return estimates


def _square_error(y, y_hat):
    return (y - y_hat)**2


def _plot_mean_square_error():
    estimates = _plot_sequence_estimate()
    sq_err = []
    for i in range(100):
        err = (_square_error(estimates[i][0], 0) + _square_error(estimates[i][1], 0) + _square_error(estimates[i][2], 0))/3
        sq_err.append(err)
    
    plt.plot(sq_err)
    plt.xlabel('Iteration')
    plt.ylabel('Mean square error')
    plt.show()



# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    X = []
    Ik = identity(k)
    cov = np.power(var, 2) * Ik
    mean = np.linspace(start_mean, end_mean, n)
    for i in range(n):
        X.append(np.random.multivariate_normal(mean[i], cov))

    return np.array(X)  


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    ...

def main():
    _plot_mean_square_error()

if __name__ == '__main__':
    main()