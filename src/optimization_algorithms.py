"""
@author: Simone Gayed Said
@author: Pierpasquale Colagrande
"""

import numpy as np
from src.ui import print_head, print_iteration, print_found_minimum

# Unused import but necessary
from mpl_toolkits.mplot3d import Axes3D


def gd(f, theta, gradient, num_iterations=5000, alpha=0.001):
    algorithm = "GD"
    print_head(algorithm, {'num_iterations': num_iterations, 'alpha': alpha})

    x_data, y_data, z_data = [theta[0]], [theta[1]], [f(theta[0], theta[1])]
    g = np.zeros(shape=2)
    t = 0
    while t < num_iterations:
        t = t + 1
        g[0] = gradient['x'](theta[0], theta[1])
        g[1] = gradient['y'](theta[0], theta[1])
        theta = theta - (alpha * g)
        x_data.append(theta[0])
        y_data.append(theta[1])
        z_data.append(f(theta[0], theta[1]))
        if t % (num_iterations / 10) == 0:
            print_iteration(theta, t)
    print_found_minimum(theta, t)
    return x_data, y_data, z_data


def adam(f, theta, gradient, num_iterations=5000, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    algorithm = "Adam"
    print_head(algorithm, {'num_iterations': num_iterations, 'alpha': alpha, 'beta_1': beta_1, 'beta_2': beta_2,
                           'epsilon': epsilon})

    x_data, y_data, z_data = [theta[0]], [theta[1]], [f(theta[0], theta[1])]
    m = np.zeros(shape=2)
    v = np.zeros(shape=2)
    g = np.zeros(shape=2)
    t = 0
    while t < num_iterations:
        t = t + 1
        g[0] = gradient['x'](theta[0], theta[1])
        g[1] = gradient['y'](theta[0], theta[1])
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        m_hat = m / (1 - np.power(beta_1, t))
        v_hat = v / (1 - np.power(beta_2, t))
        theta = theta - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        x_data.append(theta[0])
        y_data.append(theta[1])
        z_data.append(f(theta[0], theta[1]))
        if t % (num_iterations / 10) == 0:
            print_iteration(theta, t)
    print_found_minimum(theta, t)
    return x_data, y_data, z_data


def adamax(f, theta, gradient, num_iterations=5000, alpha=0.001, beta_1=0.9, beta_2=0.999):
    algorithm = "AdaMax"
    print_head(algorithm, {'num_iterations': num_iterations, 'alpha': alpha, 'beta_1': beta_1, 'beta_2': beta_2})

    x_data, y_data, z_data = [theta[0]], [theta[1]], [f(theta[0], theta[1])]
    m = np.zeros(shape=2)
    v = np.zeros(shape=2)
    g = np.zeros(shape=2)
    t = 0
    while t < num_iterations:
        t = t + 1
        g[0] = gradient['x'](theta[0], theta[1])
        g[1] = gradient['y'](theta[0], theta[1])
        m = beta_1 * m + (1 - beta_1) * g
        m_hat = m / (1 - np.power(beta_1, t))
        v = np.maximum(beta_2 * v, np.abs(g))
        theta = theta - alpha * m_hat / v
        x_data.append(theta[0])
        y_data.append(theta[1])
        z_data.append(f(theta[0], theta[1]))
        if t % (num_iterations / 10) == 0:
            print_iteration(theta, t)
    print_found_minimum(theta, t)
    return x_data, y_data, z_data


def nadam(f, theta, gradient, num_iterations=5000, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    algorithm = "Nadam"
    print_head(algorithm, {'num_iterations': num_iterations, 'alpha': alpha, 'beta_1': beta_1, 'beta_2': beta_2,
                           'epsilon': epsilon})

    x_data, y_data, z_data = [theta[0]], [theta[1]], [f(theta[0], theta[1])]
    m = np.zeros(shape=2)
    v = np.zeros(shape=2)
    g = np.zeros(shape=2)
    t = 0
    while t < num_iterations:
        t = t + 1
        g[0] = gradient['x'](theta[0], theta[1])
        g[1] = gradient['y'](theta[0], theta[1])
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        m_hat = m / (1 - np.power(beta_1, t)) + (1 - beta_1) * g / (1 - np.power(beta_1, t))
        v_hat = v / (1 - np.power(beta_2, t))
        theta = theta - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        x_data.append(theta[0])
        y_data.append(theta[1])
        z_data.append(f(theta[0], theta[1]))
        if t % (num_iterations / 10) == 0:
            print_iteration(theta, t)
    print_found_minimum(theta, t)
    return x_data, y_data, z_data


def amsgrad(f, theta, gradient, num_iterations=5000, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    algorithm = "AMSGrad"
    print_head(algorithm, {'num_iterations': num_iterations, 'alpha': alpha, 'beta_1': beta_1, 'beta_2': beta_2,
                           'epsilon': epsilon})

    x_data, y_data, z_data = [theta[0]], [theta[1]], [f(theta[0], theta[1])]
    m = np.zeros(shape=2)
    v = np.zeros(shape=2)
    v_hat = np.zeros(shape=2)
    g = np.zeros(shape=2)
    t = 0
    while t < num_iterations:
        t = t + 1
        g[0] = gradient['x'](theta[0], theta[1])
        g[1] = gradient['y'](theta[0], theta[1])
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        v_hat = np.maximum(v, v_hat)
        theta = theta - alpha * m / (np.sqrt(v_hat) + epsilon)
        x_data.append(theta[0])
        y_data.append(theta[1])
        z_data.append(f(theta[0], theta[1]))
        if t % (num_iterations / 10) == 0:
            print_iteration(theta, t)
    print_found_minimum(theta, t)
    return x_data, y_data, z_data
