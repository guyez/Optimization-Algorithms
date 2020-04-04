"""
@author: Simone Gayed Said
@author: Pierpasquale Colagrande
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def print_head(algorithm, params):
    print("\n" * 2 + " " * (50 - int(len(algorithm) / 2)) + algorithm)
    print("*" * 100)
    if 'alpha' in params:
        print("Learning Rate (alpha):", str(params['alpha']))
    if 'num_iterations' in params:
        print("Maximum Iterations:", str(params['num_iterations']))
    if 'beta_1' in params:
        print("Beta 1:", str(params['beta_1']))
    if 'beta_2' in params:
        print("Beta 2:", str(params['beta_2']))
    if 'epsilon' in params:
        print("Epsilon:", str(params['epsilon']))
    print()


def print_iteration(theta, iteration):
    print(
        "Iteration #" + str(iteration) + ": local minimum occurs at  ({:0.4f}, {:0.4f})".format(theta[0], theta[1]))


def print_introduction(function, minimum):
    print("\n" * 2 + " " * (50 - int(len("OPTIMIZATION ALGORITHMS DEMO") / 2)) + "OPTIMIZATION ALGORITHMS DEMO")
    goal = "Goal is to minimize a function using different optimization algorithms"
    print(" " * (50 - int(len(goal) / 2)) + goal + "\n")
    function = "Currently using: " + function + " function"
    print(" " * (50 - int(len(function) / 2)) + function + "\n")
    for i in range(len(minimum['X'])):
        global_min = "Having minimum in {:0.4f}, {:0.4f}".format(minimum['X'][i], minimum['Y'][i])
        print(" " * (50 - int(len(global_min) / 2)) + global_min)


def print_found_minimum(theta, iteration):
    print("\nFound minimum at ({:0.4f}, {:0.4f}) after {} iterations.".format(theta[0], theta[1], iteration))
    print("*" * 100 + "\n")


def plot_3d_minimization_procedure(X, Y, Z, x_data, y_data, z_data, minimum):
    def init():
        line_gd.set_data([], [])
        line_gd.set_3d_properties([])

        line_adam.set_data([], [])
        line_adam.set_3d_properties([])

        line_adamax.set_data([], [])
        line_adamax.set_3d_properties([])

        line_nadam.set_data([], [])
        line_nadam.set_3d_properties([])

        line_amsgrad.set_data([], [])
        line_amsgrad.set_3d_properties([])

        return line_gd, line_adam, line_adamax, line_nadam, line_amsgrad,

    def animate(i):
        line_gd.set_data(x_data['gd'][:i][::-1], y_data['gd'][:i][::-1])
        line_gd.set_3d_properties(z_data['gd'][:i][::-1])

        line_adam.set_data(x_data['adam'][:i][::-1], y_data['adam'][:i][::-1])
        line_adam.set_3d_properties(z_data['adam'][:i][::-1])

        line_adamax.set_data(x_data['adamax'][:i][::-1], y_data['adamax'][:i][::-1])
        line_adamax.set_3d_properties(z_data['adamax'][:i][::-1])

        line_nadam.set_data(x_data['nadam'][:i][::-1], y_data['nadam'][:i][::-1])
        line_nadam.set_3d_properties(z_data['nadam'][:i][::-1])

        line_amsgrad.set_data(x_data['amsgrad'][:i][::-1], y_data['amsgrad'][:i][::-1])
        line_amsgrad.set_3d_properties(z_data['amsgrad'][:i][::-1])

        return line_gd, line_adam, line_adamax, line_nadam, line_amsgrad,

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    fig.canvas.set_window_title("Descent Animation - 3D View")
    ax.set_title("Descent Animation - 3D View")
    ax.view_init(azim=290)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color="grey", edgecolor='none', alpha=0.25)

    ax.scatter(minimum['X'], minimum['Y'], minimum['Z'], marker="*", color="red", alpha=0.7, s=150)

    line_gd, = ax.plot([], [], [], "-o", lw=2, color='blue', markevery=10000)
    line_adam, = ax.plot([], [], [], "-o", lw=2, color='green', markevery=10000)
    line_adamax, = ax.plot([], [], [], "-o", lw=2, color='magenta', markevery=10000)
    line_nadam, = ax.plot([], [], [], "-o", lw=2, color='yellow', markevery=10000)
    line_amsgrad, = ax.plot([], [], [], "-o", lw=2, color='orange', markevery=10000)

    plt.legend((line_gd, line_adam, line_adamax, line_nadam, line_amsgrad),
               ('GD', 'ADAM', 'ADAMAX', 'NADAM', 'AMSGRAD'))
    animation.FuncAnimation(fig, animate, init_func=init, frames=5000, interval=1, blit=True, repeat=False)

    plt.show()


def plot_2d_minimization_procedure(X, Y, Z, x_data, y_data, minimum):
    def init():
        line_gd.set_data([], [])
        line_adam.set_data([], [])
        line_adamax.set_data([], [])
        line_nadam.set_data([], [])
        line_amsgrad.set_data([], [])

        return line_gd, line_adam, line_adamax, line_nadam, line_amsgrad,

    def animate(i):
        line_gd.set_data(x_data['gd'][:i][::-1], y_data['gd'][:i][::-1])

        line_adam.set_data(x_data['adam'][:i][::-1], y_data['adam'][:i][::-1])

        line_adamax.set_data(x_data['adamax'][:i][::-1], y_data['adamax'][:i][::-1])

        line_nadam.set_data(x_data['nadam'][:i][::-1], y_data['nadam'][:i][::-1])

        line_amsgrad.set_data(x_data['amsgrad'][:i][::-1], y_data['amsgrad'][:i][::-1])

        return line_gd, line_adam, line_adamax, line_nadam, line_amsgrad,

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('Descent Animation - Top View')
    ax.set_title("Descent Animation - Top View")

    ax.scatter(minimum['X'], minimum['Y'], marker="*", color="red", alpha=0.7, s=150)

    line_gd, = ax.plot([], [], "-o", lw=2, color='blue', markevery=10000)
    line_adam, = ax.plot([], [], "-o", lw=2, color='green', markevery=10000)
    line_adamax, = ax.plot([], [], "-o", lw=2, color='magenta', markevery=10000)
    line_nadam, = ax.plot([], [], "-o", lw=2, color='yellow', markevery=10000)
    line_amsgrad, = ax.plot([], "-o", lw=2, color='orange', markevery=10000)

    CS = plt.contour(X, Y, Z, colors=['grey'])
    plt.clabel(CS, inline=1, fontsize=10)

    plt.legend((line_gd, line_adam, line_adamax, line_nadam, line_amsgrad),
               ('GD', 'ADAM', 'ADAMAX', 'NADAM', 'AMSGRAD'))
    animation.FuncAnimation(fig, animate, init_func=init, frames=5000, interval=1, blit=True, repeat=False)

    plt.show()


def plot_graph(data, axis='x'):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title(axis.upper() + " axis")
    ax.set_title(axis.upper() + " axis")

    plt.plot(data['gd'], color="blue")
    plt.plot(data['adam'], color="green")
    plt.plot(data['adamax'], color="magenta")
    plt.plot(data['nadam'], color="yellow")
    plt.plot(data['amsgrad'], color="orange")
    plt.ylabel(axis)
    plt.xlabel('iterations')
    plt.legend(['GD', 'ADAM', '', 'NADAM', 'AMSGRAD'], loc='upper right')
    plt.show()
