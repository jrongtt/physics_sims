import numpy as np
from numpy import pi
from scipy.constants import hbar
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Constants for the simulation
m = 9.10938356e-31  # mass of electron (kg)
L = 10e-9  # domain size in meters
N = 1024  # number of spatial points
dx = L / N  # spatial resolution
x = np.linspace(-L/2, L/2, N)  # spatial domain
dt = 1e-17  # time step in seconds
T = 1e-15  # total time of simulation in seconds
frames = int(T / dt)  # number of frames in the animation

# Initial wave packet parameters
x0 = 0
sigma = 1e-10
k0 = 5e10

# Initial wave function: Gaussian wave packet
psi_x = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi_x /= np.sqrt(np.sum(np.abs(psi_x)**2) * dx)  # normalization

# Prepare momentum space operations
p = np.fft.fftfreq(N, d=dx) * 2 * pi * hbar  # momentum array
energy_p = p**2 / (2 * m)  # kinetic energy in momentum space
exp_factor = np.exp(-1j * energy_p * dt / hbar)  # exponential factor for time evolution

# Function to update wave function for each frame
def update_psi(psi_x):
    psi_p = np.fft.fft(psi_x)  # transform to momentum space
    psi_p *= exp_factor  # apply the time evolution operator
    psi_x = np.fft.ifft(psi_p)  # transform back to position space
    return psi_x

# Generate frames and save as video
output_path = 'C:\\Users\\jackp\\quantum_simulations\\free_particle_simulation.mp4'
writer = imageio.get_writer(output_path, fps=30)

for i in range(frames):
    psi_x = update_psi(psi_x)
    fig_width, fig_height = 800, 480  # Ensure dimensions are divisible by 16
    plt.figure(figsize=(fig_width / 100, fig_height / 100))
    plt.plot(x, np.abs(psi_x)**2, color='blue')
    plt.xlim(-L/2, L/2)
    plt.ylim(0, 1.2 * np.max(np.abs(psi_x)**2))
    plt.xlabel('x (meters)')
    plt.ylabel('Probability Density')
    plt.title(f'Time = {i*dt:.2e} seconds')
    plt.grid(True)

    # Save the current frame
    plt.savefig('frame.png')
    plt.close()
    writer.append_data(imageio.imread('frame.png'))

writer.close()
print('Animation saved to:', output_path)

