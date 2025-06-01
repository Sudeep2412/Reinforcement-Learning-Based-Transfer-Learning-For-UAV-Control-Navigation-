import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

# Simulated transfer function (replace with system identification results)
s = ctrl.TransferFunction.s
G1_x = (4**2) / (s**2 + 2*0.7*4*s + 4**2)  # Source UAV
G2_x = (1**2) / (s**2 + 2*0.8*1*s + 1**2)  # Target UAV

# Plot step response to visualize dynamics
time = np.linspace(0, 10, 1000)
t, y1 = ctrl.step_response(G1_x, time)
t, y2 = ctrl.step_response(G2_x, time)

plt.plot(t, y1, label="Source UAV (G1)")
plt.plot(t, y2, label="Target UAV (G2)")
plt.xlabel("Time")
plt.ylabel("Response")
plt.legend()
plt.title("System Response of UAVs")
plt.show()
