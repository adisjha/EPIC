#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('orbit_data.csv', delimiter=',', skip_header=1)

x1, y1, x2, y2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
vx1, vy1, vx2, vy2 = data[:, 4], data[:, 5], data[:, 6], data[:, 7]
Fx1, Fy1, Fx2, Fy2 = data[:, 8], data[:, 9], data[:, 10], data[:, 11]
Px, Py, t = data[:, 12], data[:, 13], data[:, 14]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.plot(x1, y1, 'b-', linewidth=0.5, label='Particle 1 (-q)')
ax1.plot(x2, y2, 'r-', linewidth=0.5, label='Particle 2 (+q)')
ax1.plot(x1[0], y1[0], 'bo', markersize=8, label='Start 1')
ax1.plot(x2[0], y2[0], 'ro', markersize=8, label='Start 2')
ax1.plot(x1[-1], y1[-1], 'bx', markersize=10, label='End 1')
ax1.plot(x2[-1], y2[-1], 'rx', markersize=10, label='End 2')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Particle Orbits')
ax1.legend(loc='best', fontsize=8)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(t, vx1, 'b-', linewidth=0.5, label='vx1')
ax2.plot(t, vy1, 'b--', linewidth=0.5, label='vy1')
ax2.plot(t, vx2, 'r-', linewidth=0.5, label='vx2')
ax2.plot(t, vy2, 'r--', linewidth=0.5, label='vy2')
ax2.set_xlabel('Time')
ax2.set_ylabel('Velocity')
ax2.set_title('Velocity vs Time')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
speed1 = np.sqrt(vx1**2 + vy1**2)
speed2 = np.sqrt(vx2**2 + vy2**2)
ax3.plot(t, speed1, 'b-', linewidth=0.5, label='|v1|')
ax3.plot(t, speed2, 'r-', linewidth=0.5, label='|v2|')
ax3.set_xlabel('Time')
ax3.set_ylabel('Speed')
ax3.set_title('Speed vs Time')
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
force_mag1 = np.sqrt(Fx1**2 + Fy1**2)
force_mag2 = np.sqrt(Fx2**2 + Fy2**2)
ax4.plot(t, force_mag1, 'b-', linewidth=0.5, label='|F1|')
ax4.plot(t, force_mag2, 'r-', linewidth=0.5, label='|F2|')
ax4.set_xlabel('Time')
ax4.set_ylabel('Force Magnitude')
ax4.set_title('Force Magnitude vs Time')
ax4.legend(loc='best', fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('orbit_plot.png', dpi=150, bbox_inches='tight')
print("Saved orbit_plot.png")

plt.figure(figsize=(8, 6))
e1 = 0.5 * (vx1**2 + vy1**2)
e2 = 0.5 * (vx2**2 + vy2**2)
e_total = e1 + e2
ax = plt.gca()
ax.plot(t, e1, 'b-', linewidth=0.5, label='KE1')
ax.plot(t, e2, 'r-', linewidth=0.5, label='KE2')
ax.plot(t, e_total, 'k-', linewidth=1, label='Total KE')
ax.set_xlabel('Time')
ax.set_ylabel('Kinetic Energy')
ax.set_title('Kinetic Energy vs Time')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('orbit_energy.png', dpi=150, bbox_inches='tight')
print("Saved orbit_energy.png")

print(f"\nSimulation summary:")
print(f"  Steps: {len(t)}")
print(f"  Time: {t[0]:.4f} to {t[-1]:.4f}")
sep_init = np.sqrt((x1[0]-x2[0])**2 + (y1[0]-y2[0])**2)
sep_final = np.sqrt((x1[-1]-x2[-1])**2 + (y1[-1]-y2[-1])**2)
print(f"  Initial separation: {sep_init:.4f}")
print(f"  Final separation: {sep_final:.4f}")
print(f"  Initial total KE: {e_total[0]:.6f}")
print(f"  Final total KE: {e_total[-1]:.6f}")
print(f"  Energy change: {(e_total[-1]-e_total[0])/e_total[0]*100:.2f}%")
