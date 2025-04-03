# Title: Animated Lorenz Attractors with Multiple Initial Conditions
#
# This code animates the Lorenz attractor using matplotlib's FuncAnimation.
# It computes the trajectory of the Lorenz system for multiple initial conditions and updates the plot in real-time.
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

# Number of trajectories
num_curves = 5  
initial_conditions = [
    (0., 1., 1.05),
    (0.1, 0.9, 1.1),
    (-0.2, -1.1, 1.0),
    (0.3, 0.8, 1.2),
    (-0.1, -0.9, 1.05)
]

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

# Create lines with different colors
colors = ['r', 'g', 'b', 'm', 'c']
lines = [ax.plot([], [], [], lw=1.5, color=colors[i])[0] for i in range(num_curves)]

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