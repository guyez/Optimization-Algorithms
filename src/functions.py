"""
@author: Simone Gayed Said
@author: Pierpasquale Colagrande
"""

import numpy as np


def sixhump_camel_function():
    f = lambda x, y: (4 - 2.1 * x ** 2 + x ** 4 / 3) * x ** 2 + x * y + (-4 + 4 * y ** 2) * y ** 2

    g_x = lambda x, y: 2 * (x ** 5 - 4.2 * x ** 3 + 4 * x + 0.5 * y)

    g_y = lambda x, y: x + 16 * y ** 3 - 8 * y

    gradient = {'x': g_x, 'y': g_y}

    x_start = 1
    y_start = 1

    theta = [x_start, y_start]

    X, Y = np.mgrid[-3:3:50j, -2:2:50j]

    minimum = {'X': [0.0898, -0.0898], 'Y': [-0.7126, 0.7126], 'Z': [-1.0316, -1.0316]}

    name = "Sixhump Camel"

    return f, X, Y, gradient, theta, minimum, name


def easom_function():
    f = lambda x, y: -np.cos(x) * np.cos(y) * np.exp(-(x - np.pi) ** 2 - (y - np.pi) ** 2)

    g_x = lambda x, y: 2 * np.exp(-(-np.pi + x) ** 2 - (-np.pi + y) ** 2) * (-np.pi + x) * np.cos(x) * np.cos(
        y) + np.exp(-(-np.pi + x) ** 2 - (-np.pi + y) ** 2) * np.cos(y) * np.sin(x)

    g_y = lambda x, y: 2 * np.exp(-(-np.pi + x) ** 2 - (-np.pi + y) ** 2) * (-np.pi + y) * np.cos(x) * np.cos(
        y) + np.exp(-(-np.pi + x) ** 2 - (-np.pi + y) ** 2) * np.cos(x) * np.sin(y)

    gradient = {'x': g_x, 'y': g_y}

    x_start = 1.7
    y_start = 3

    theta = [x_start, y_start]

    X, Y = np.mgrid[-2.5:7.5:50j, -2.5:7.5:50j]

    minimum = {'X': [np.pi], 'Y': [np.pi], 'Z': [-1]}

    name = "Easom"

    return f, X, Y, gradient, theta, minimum, name


def bukin_n6_function():
    f = lambda x, y: 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10)

    g_x = lambda x, y: (0.01 * (10 + x)) / np.abs(10 + x) - (x * (-0.01 * x ** 2 + y)) / np.abs(-0.01 * x ** 2 + y) ** (
            3 / 2)

    g_y = lambda x, y: (50 * (-0.01 * x ** 2 + y)) / np.abs(-0.01 * x ** 2 + y) ** (3 / 2)

    gradient = {'x': g_x, 'y': g_y}

    x_start = -8
    y_start = 2

    theta = [x_start, y_start]

    X, Y = np.mgrid[-15:-5:50j, -3:3:50j]

    minimum = {'X': [-10], 'Y': [1], 'Z': [0]}

    name = "Bukin N6"

    return f, X, Y, gradient, theta, minimum, name


def drop_wave_function():
    f = lambda x, y: - (1 + np.cos(12 * np.sqrt(x ** 2 + y ** 2))) / (0.5 * (x ** 2 + y ** 2) + 2)

    g_x = lambda x, y: ((12 * x * np.sin(12 * np.sqrt(x ** 2 + y ** 2))) / (
            np.sqrt(x ** 2 + y ** 2) * (0.5 * (x ** 2 + y ** 2) + 2))) - (
                               (x * (-np.cos(12 * np.sqrt(x ** 2 + y ** 2)) - 1)) / (
                               0.5 * (x ** 2 + y ** 2) + 2) ** 2)

    g_y = lambda x, y: ((12 * y * np.sin(12 * np.sqrt(x ** 2 + y ** 2))) / (
            np.sqrt(x ** 2 + y ** 2) * (0.5 * (x ** 2 + y ** 2) + 2))) - (
                               (y * (-np.cos(12 * np.sqrt(x ** 2 + y ** 2)) - 1)) / (
                               0.5 * (x ** 2 + y ** 2) + 2) ** 2)

    gradient = {'x': g_x, 'y': g_y}

    x_start = 0.5
    y_start = 0.5

    theta = [x_start, y_start]

    X, Y = np.mgrid[-2:2:50j, -2:2:50j]

    minimum = {'X': [0], 'Y': [0], 'Z': [-1]}

    name = "Dropwave"

    return f, X, Y, gradient, theta, minimum, name


def matyas_function():
    f = lambda x, y: 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

    g_x = lambda x, y: 0.52 * x - 0.48 * y

    g_y = lambda x, y: 0.52 * y - 0.48 * x

    gradient = {'x': g_x, 'y': g_y}

    x_start = -10
    y_start = 10

    theta = [x_start, y_start]

    X, Y = np.mgrid[-10:10:50j, -10:10:50j]

    minimum = {'X': [0], 'Y': [0], 'Z': [0]}

    name = "Matyas"

    return f, X, Y, gradient, theta, minimum, name
