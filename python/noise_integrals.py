import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Number of time steps
num_steps = 100

# Generate white noise
white_noise = np.random.normal(0, 1, num_steps)

# Generate random walk
random_walk = np.cumsum(white_noise)

# Generate random walk with drift
drift = np.mean(white_noise)  # Drift parameter
random_walk_with_drift = np.cumsum(white_noise + drift)

# Time array
time = np.arange(0, num_steps)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(time, white_noise, label=f'White Noise')
plt.title('White Noise with mean={round(drift,4)}')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, random_walk, label='Random Walk')
plt.title('Random Walk')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, random_walk_with_drift, label='Random Walk with Drift')
plt.title('Random Walk with Drift (using mean from White Noise')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()
