import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# parameters
k = 1  # wave number
x = np.linspace(0, 20, 500) 

# real and imaginary parts
real_part = np.cos(k * x)
imag_part = np.sin(k * x)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# rendered line
line, = ax.plot([], [], [], lw=2)


ax.set_xlim(0, 20)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('Real')
ax.set_zlabel('Imaginary')


def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,


def animate(i):
    line.set_data(x[:i], real_part[:i])
    line.set_3d_properties(imag_part[:i])
    return line,


ani = FuncAnimation(fig, animate, init_func=init, frames=len(x), interval=30, blit=True)
ani.save('complex_exponential_helix.gif', writer='pillow', fps=30)

plt.show()

