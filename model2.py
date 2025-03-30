##=========A simple chaotic stock price model based on a modified Lorenz system======

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
