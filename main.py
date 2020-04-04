"""
@author: Simone Gayed Said
@author: Pierpasquale Colagrande
"""

from src.ui import print_introduction, plot_graph, plot_3d_minimization_procedure, \
    plot_2d_minimization_procedure  # ,plot_3d_surface
from src.optimization_algorithms import gd, adam, adamax, nadam, amsgrad
from src.functions import sixhump_camel_function, easom_function, bukin_n6_function, drop_wave_function, matyas_function

# f, X, Y, gradient, theta, minimum, name = sixhump_camel_function()

f, X, Y, gradient, theta, minimum, name = easom_function()

# f, X, Y, gradient, theta, minimum, name = bukin_n6_function()

# f, X, Y, gradient, theta, minimum, name = drop_wave_function()

# f, X, Y, gradient, theta, minimum, name = matyas_function()

print_introduction(name, minimum)

Z = f(X, Y)

x_data_gd, y_data_gd, z_data_gd = gd(f, theta, gradient)
x_data_adam, y_data_adam, z_data_adam = adam(f, theta, gradient)
x_data_adamax, y_data_adamax, z_data_adamax = adamax(f, theta, gradient)
x_data_nadam, y_data_nadam, z_data_nadam = nadam(f, theta, gradient)
x_data_amsgrad, y_data_amsgrad, z_data_amsgrad = amsgrad(f, theta, gradient)

x_data = {'gd': x_data_gd, 'adam': x_data_adam, 'adamax': x_data_adamax, 'nadam': x_data_nadam,
          'amsgrad': x_data_amsgrad}

y_data = {'gd': y_data_gd, 'adam': y_data_adam, 'adamax': y_data_adamax, 'nadam': y_data_nadam,
          'amsgrad': y_data_amsgrad}

z_data = {'gd': z_data_gd, 'adam': z_data_adam, 'adamax': z_data_adamax, 'nadam': z_data_nadam,
          'amsgrad': z_data_amsgrad}

plot_3d_minimization_procedure(X, Y, Z, x_data, y_data, z_data, minimum)

plot_2d_minimization_procedure(X, Y, Z, x_data, y_data, minimum)

plot_graph(x_data)
plot_graph(y_data, 'y')
