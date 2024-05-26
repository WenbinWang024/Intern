import numpy as np

def negentropy_prox(x, grad, eta):
    """ Proximal operator for the negentropy over simplex. """
    return np.exp(np.log(x) - eta * grad) / np.sum(np.exp(np.log(x) - eta * grad))

def sp_mp_matrix_game(A, num_iterations, eta):
    n, m = A.shape
    x = np.ones(n) / n
    y = np.ones(m) / m

    for _ in range(num_iterations):
        # Compute gradients
        Ay = A @ y
        xTA = x @ A

        # Proximal updates
        x_new = negentropy_prox(x, Ay, eta)
        y_new = negentropy_prox(y, -xTA, eta)

        # Update strategies
        x, y = x_new, y_new

    return x, y

# Loss matrix A for the zero-sum game (example matrix)
A = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])

# Parameters
num_iterations = 1000
eta = 0.1

# Running the SP-MP algorithm
x_opt, y_opt = sp_mp_matrix_game(A, num_iterations, eta)
print("Optimal strategy for player 1:", x_opt)
print("Optimal strategy for player 2:", y_opt)