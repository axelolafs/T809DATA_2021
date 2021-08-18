import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    t = np.exp(-(np.power((x-mu),2))/(2*np.power(sigma, 2)))
    n = np.sqrt(2*np.pi*np.power(sigma, 2))
    p = t/n
    return p
def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    x_range = np.linspace(x_start, x_end, 500)
    distribution = normal(x_range, sigma, mu)
    plt.plot(x_range, distribution)

def _plot_three_normals():
    # Part 1.2
    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1, 1.5, -5, 5)
    plt.show()

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    n = len(weights)
    gaussianSum = 0
    for i in range(n):
        density_i = weights[i] * normal(x, sigmas[i], mus[i])
        gaussianSum += density_i
    return gaussianSum

def _compare_components_and_mixture():
    # Part 2.2
    x_start = -5
    x_end = 5
    weights = [1/3, 1/3, 1/3]
    mus = [0, -0.5, 1.5]
    sigmas = [0.5, 1.5, 0.25]
    plot_normal(sigmas[0], mus[0], x_start, x_end)
    plot_normal(sigmas[1], mus[1], x_start, x_end)
    plot_normal(sigmas[2], mus[2], x_start, x_end)

    x_range = np.linspace(x_start, x_end, 500)
    distribution = normal_mixture(x_range, sigmas, mus, weights)

    plt.plot(x_range, distribution)
    plt.show()


def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1
    return 1
def _plot_mixture_and_samples():
    # Part 3.2
    return 1
if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    _plot_three_normals()
    _compare_components_and_mixture()