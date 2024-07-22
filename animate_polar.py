import numpy as np
import matplotlib.pyplot as plt

# Constants
a = 0.6  # Adjust as necessary
phi = np.pi / 4  # Phase, adjust as necessary

# Function to compute the modulus squared of the dot product
def probability(theta):
    # Vector from eigenvector
    vector_psi = np.array([a, np.sqrt(1 - a**2) * np.exp(1j * phi)])
    
    # Vector from angles
    vector_theta = np.array([np.cos(theta), np.sin(theta)])
    
    # Dot product
    dot_product = np.dot(vector_psi, vector_theta)
    
    # Modulus squared of the complex dot product
    return np.abs(dot_product)**2

# Theta values from 0 to 2*pi
theta_values = np.linspace(0, 2 * np.pi, 1000)
prob_values = np.vectorize(probability)(theta_values)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(theta_values, prob_values, label='Probability')
plt.title('Probability as a Function of Theta')
plt.xlabel('Theta (radians)')
plt.ylabel('Probability')
plt.grid(True)
plt.axvline(x=theta_values[np.argmax(prob_values)], color='r', linestyle='--', label=f'Max at Theta = {theta_values[np.argmax(prob_values)]:.2f} radians')
plt.legend()
plt.show()

