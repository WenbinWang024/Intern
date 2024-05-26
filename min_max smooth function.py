# Here is a Python implementation of the SP-MP algorithm for solving the problem where you want to minimize the maximum of several smooth functions, which are L-Lipschitz and β-smooth

import numpy as np

def grad_fi(x, i):
    """ Gradient of the i-th function at x """
    # Example: return the gradient of the i-th function, this needs to be defined based on the specific problem
    return np.array([2 * (x[0] - i), 2 * (x[1] - i)])

def phi(x, y):
    """ The function phi(x, y) = <~f(x), y> """
    # Compute ~f(x) for given x
    f_x = np.array([fi(x) for fi in [lambda x: (x[0] - 1)**2, lambda x: (x[1] - 2)**2]])
    return f_x @ y

def mirror_prox_method(x0, num_iterations, eta):
    x = x0
    y = np.array([1/2, 1/2])  # Simplex initialization
    for _ in range(num_iterations):
        # Gradient calculation for current x, y
        grad_x_phi = sum(yi * grad_fi(x, i) for i, yi in enumerate(y))
        grad_y_phi = np.array([fi(x) for fi in [lambda x: (x[0] - 1)**2, lambda x: (x[1] - 2)**2]])
        
        # Mirror descent update for y
        y = y * np.exp(-eta * grad_y_phi)
        y /= np.sum(y)  # Project back to simplex
        
        # Mirror descent update for x
        x = x - eta * grad_x_phi
        
    return x, phi(x, y)

# Initial point and parameters
x0 = np.array([0.0, 0.0])
num_iterations = 100
eta = 0.01

# Run the Mirror Prox method
opt_x, opt_value = mirror_prox_method(x0, num_iterations, eta)
print("Optimal x:", opt_x)
print("Optimal value:", opt_value)