# import matplotlib.pyplot as plt
# import numpy as np


# def lorenz(xyz, *, s=10, r=28, b=2.667):
#     """
#     Parameters
#     ----------
#     xyz : array-like, shape (3,)
#        Point of interest in three-dimensional space.
#     s, r, b : float
#        Parameters defining the Lorenz attractor.

#     Returns
#     -------
#     xyz_dot : array, shape (3,)
#        Values of the Lorenz attractor's partial derivatives at *xyz*.
#     """
#     x, y, z = xyz
#     x_dot = s*(y - x)
#     y_dot = r*x - y - x*z
#     z_dot = x*y - b*z
#     return np.array([x_dot, y_dot, z_dot])


# dt = 0.01
# num_steps = 10000

# xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
# xyzs[0] = (0., 1., 1.05)  # Set initial values
# # Step through "time", calculating the partial derivatives at the current point
# # and using them to estimate the next point
# for i in range(num_steps):
#     xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

# # Plot
# ax = plt.figure().add_subplot(projection='3d')

# ax.plot(*xyzs.T, lw=0.5)
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_title("Lorenz Attractor")

# plt.show()

###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


# Time step and number of iterations
dt = 0.01
num_steps = 5000  # Reduced for smoother animation

# Initialize array for points
xyzs = np.empty((num_steps + 1, 3))
xyzs[0] = (0., 1., 1.05)

# Compute Lorenz attractor points
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim([-20, 20])
ax.set_ylim([-30, 30])
ax.set_zlim([0, 50])

ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor Animation")

# Initialize line object
line, = ax.plot([], [], [], lw=1)


# Update function for animation
def update(num):
    line.set_data(xyzs[:num, 0], xyzs[:num, 1])
    line.set_3d_properties(xyzs[:num, 2])
    return line,


# Animate using FuncAnimation
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=1, blit=True)

plt.show()

#####################################################################################

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as animation

# # Lorenz system function
# def lorenz(xyz, s=10, r=28, b=2.667):
#     x, y, z = xyz
#     x_dot = s * (y - x)
#     y_dot = r * x - y - x * z
#     z_dot = x * y - b * z
#     return np.array([x_dot, y_dot, z_dot])

# # Simulation parameters
# dt = 0.01
# num_steps = 5000  # Animation frames

# # Number of trajectories
# num_curves = 5  
# initial_conditions = [
#     (0., 1., 1.05),
#     (0.1, 0.9, 1.1),
#     (-0.2, -1.1, 1.0),
#     (0.3, 0.8, 1.2),
#     (-0.1, -0.9, 1.05)
# ]

# # Generate trajectories
# trajectories = []
# for init in initial_conditions:
#     xyzs = np.empty((num_steps + 1, 3))
#     xyzs[0] = init
#     for i in range(num_steps):
#         xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
#     trajectories.append(xyzs)

# # Set up the figure
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_xlim([-20, 20])
# ax.set_ylim([-30, 30])
# ax.set_zlim([0, 50])

# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_title("Animated Lorenz Attractors")

# # Create lines with different colors
# colors = ['r', 'g', 'b', 'm', 'c']
# lines = [ax.plot([], [], [], lw=1.5, color=colors[i])[0] for i in range(num_curves)]

# # Update function for animation
# def update(num):
#     for i, line in enumerate(lines):
#         xyzs = trajectories[i]
#         line.set_data(xyzs[:num, 0], xyzs[:num, 1])
#         line.set_3d_properties(xyzs[:num, 2])
#     return lines

# # Animate
# ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=1, blit=True)

# plt.show()

############################################################################


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def stock_market(t, xyz, alpha=0.1, beta=0.2, gamma=0.3):
    """
    A simple chaotic stock price model based on a modified Lorenz system.
    """
    x, y, z = xyz
    dx_dt = alpha * (y - x)  # Trend following behavior
    dy_dt = beta * (x - y - x*z)  # Market reaction with external effects
    dz_dt = gamma * (x*y - z)  # Long-term market behavior
    return [dx_dt, dy_dt, dz_dt]

# Time range
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

# Initial conditions
initial_conditions = [1, 1, 1.5]

# Solve the system
sol = solve_ivp(stock_market, t_span, initial_conditions, t_eval=t_eval)

# Plot the results
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.7, color='blue')
ax.set_xlabel("Stock Price (X)")
ax.set_ylabel("Market Sentiment (Y)")
ax.set_zlabel("External Factors (Z)")
ax.set_title("Chaotic Stock Market Simulation")
plt.show()


