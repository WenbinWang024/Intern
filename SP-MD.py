# pseudocode for saddle point mirror descent
# input: initial point z0, learning rate eta, number of iterations T
# output: final point zT


# z0 = (x0, y0) initial point
# for t = 1, 2, ..., T do
#     g_t = (g_{X, t}, g_{Y, t}) compute the gradient at z_t = (x_t, y_t)
#       g_{X, t} is the gradient with respect to x, g_{Y, t} is the gradient with respect to y

# Update z_{t+1} = (x_{t+1}, y_{t+1}) as follows:
#     z_{t+1} = argmin_{z \in Z & D} (eta * <g_t, z> + D_phi(z, z_t))
#       where D_phi is the Bregman divergence, phi(z) = a * phi_X(x) + b * phi_Y(y)