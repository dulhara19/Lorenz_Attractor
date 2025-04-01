import matplotlib.pyplot as plt
import numpy as np


def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])  #


dt = 0.01
num_steps = 10000

xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
xyzs[0] = (0., 1., 1.05)  # Set initial values
# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

# Plot
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*xyzs.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

# ###############################################################################

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

########################################################################

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Lorenz system function
def lorenz(xyz, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])

# Simulation parameters
dt = 0.01
num_steps = 5000  # Animation frames

# Number of trajectories (1000 initial conditions)
num_curves = 1000  
initial_conditions = np.random.uniform(low=-2, high=2, size=(num_curves, 3))  # Generate 1000 random initial values in the range [-2, 2]

# Generate trajectories
trajectories = []
for init in initial_conditions:
    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = init
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
    trajectories.append(xyzs)

# Set up the figure
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim([-20, 20])
ax.set_ylim([-30, 30])
ax.set_zlim([0, 50])

ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Animated Lorenz Attractors")

# Create lines for the curves
colors = plt.cm.viridis(np.linspace(0, 1, num_curves))  # Use a colormap for a smooth color gradient
lines = [ax.plot([], [], [], lw=0.5, color=colors[i])[0] for i in range(num_curves)]

# Update function for animation
def update(num):
    for i, line in enumerate(lines):
        xyzs = trajectories[i]
        line.set_data(xyzs[:num, 0], xyzs[:num, 1])
        line.set_3d_properties(xyzs[:num, 2])
    return lines

# Animate
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=1, blit=True)

plt.show()
