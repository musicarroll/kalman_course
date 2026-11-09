# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:38:20 2023

@author: cognotrend
"""


import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def cannonball_trajectory(v0, angle, g=9.8):
    """
    Simulates the trajectory of a cannonball.
    
    Arguments:
    - v0: Initial velocity of the cannonball (in meters per second).
    - angle: Launch angle of the cannonball (in degrees).
    - g: Acceleration due to gravity (in meters per second squared). Default is 9.8 m/s^2.
    
    Returns:
    - x: List of x-coordinates of the cannonball at different time points.
    - y: List of y-coordinates of the cannonball at different time points.
    """
    
    # Convert the angle from degrees to radians
    angle_rad = math.radians(angle)
    
    # Calculate the initial horizontal and vertical velocities
    v0x = v0 * math.cos(angle_rad)
    v0y = v0 * math.sin(angle_rad)
    
    # Calculate the total time of flight
    t_total = (2 * v0y) / g
    
    # Calculate the time intervals
    t_intervals = []
    t = 0.0
    dt = t_total / 100  # Splitting the time into 100 intervals
    while t <= t_total:
        t_intervals.append(t)
        t += dt
    
    # Calculate the x and y coordinates for each time interval
    x = []
    y = []
    for t in t_intervals:
        x.append(v0x * t)
        y.append(v0y * t - 0.5 * g * t**2)
    
    return x, y

def add_noise(value, noise_std):
    """
    Adds additive white Gaussian noise to a value.
    
    Arguments:
    - value: The original value.
    - noise_std: The standard deviation of the Gaussian noise.
    
    Returns:
    - The noisy value.
    """
    noise = random.gauss(0, noise_std)
    return value + noise

def update(frame):
    """
    Update function for the animation.
    
    Arguments:
    - frame: Current frame index.
    """
    plt.cla()  # Clear the current plot
    
    # Plot the true trajectory
    plt.plot(x_coordinates, y_coordinates, label='True Trajectory')
    
    # Plot the measured position
    x = x_coordinates[frame]
    y = y_coordinates[frame]
    plt.scatter(x, y, color='red', label='Measured Position')
    
    plt.title('Cannonball Trajectory')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Distance (m)')
    plt.legend()
    plt.grid(True)

# Example usage
initial_velocity = 50.0  # m/s
launch_angle = 45.0  # degrees
angle_sensor_noise_std = 2.0  # Standard deviation of the sensor noise
range_sensor_noise_std = 10.0  # Standard deviation of the sensor noise

x_coordinates, y_coordinates = cannonball_trajectory(initial_velocity, launch_angle)

# Simulate sensor measurements
sensor_angles = []
sensor_ranges = []
for i in range(len(x_coordinates)):
    x = x_coordinates[i]
    y = y_coordinates[i]
    angle = math.atan2(y, x)  # Compute the angle to the cannonball
    range_ = math.sqrt(x**2 + y**2)  # Compute the range to the cannonball
    angle_noisy = add_noise(angle, angle_sensor_noise_std)  # Add noise to the angle measurement
    range_noisy = add_noise(range_, range_sensor_noise_std)  # Add noise to the range measurement
    sensor_angles.append(angle_noisy)
    sensor_ranges.append(range_noisy)

# Set up the figure and axis for animation
fig, ax = plt.subplots()
plt.grid(True)

# Create the animation
animation = FuncAnimation(fig, update, frames=len(x_coordinates), interval=200)

plt.show()
