#______ Animated Lorenz Attractor _______
# This code animates the Lorenz attractor using matplotlib's FuncAnimation.
# It computes the trajectory of the Lorenz system and updates the plot in real-time.
#
# The Lorenz system is a set of ordinary differential equations that model chaotic behavior.
# The parameters s, r, and b define the system's dynamics.
# The animation shows the trajectory in 3D space, illustrating the chaotic nature of the system.
#
# The code uses numpy for numerical calculations and matplotlib for plotting and animation.
# It initializes the figure, sets axis limits, and creates a line object for the trajectory.
# The update function modifies the line data for each frame of the animation.
# Finally, it uses FuncAnimation to create the animation and display it.
#
#
# Import necessary libraries
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