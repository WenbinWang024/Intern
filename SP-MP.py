# pseudocode for saddle point mirror prox
# input: initial point z0, learning rate eta, number of iterations T
# output: final point zT


# z0 initial point
# for t = 1, 2, ..., T do
#     w_{t+1} = argmin_{z \in Z & D} (eta * <(∇_x φ(x_t, y_t), -∇_y φ(x_t, y_t)), z>) + + D_Φ(z, z_t))
#     z_{t+1} = argmin_{z \in Z & D} (eta * <(∇_x φ(u_{t+1}, v_{t+1}), -∇_y φ(u_{t+1}, v_{t+1})), z>) + D_Φ(z, z_t)