import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dt = 1.0  # Time step

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_covariance, measurement_noise_covariance):
        self.state = np.array(initial_state)
        self.covariance = np.array(initial_covariance)
        self.covariance_predicted = self.covariance
        self.state_predicted = self.state
        self.process_noise_covariance = np.array(process_noise_covariance)
        self.measurement_noise_covariance = np.array(measurement_noise_covariance)

    def predict(self, A):
        self.state_predicted = np.dot(A, self.state)
        self.covariance_predicted = np.dot(A, np.dot(self.covariance, A.T)) + self.process_noise_covariance
        return self.state_predicted

    def update(self, measurement, H):
        K = np.dot(self.covariance_predicted, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(self.covariance_predicted, H.T)) + self.measurement_noise_covariance)))
        innovation = measurement - np.dot(H, self.state_predicted)
        self.state = self.state_predicted + np.dot(K, innovation)
        
        I = np.eye(self.state.shape[0])
        self.covariance = np.dot((I - np.dot(K, H)), np.dot(self.covariance_predicted, (I - np.dot(K, H)).T)) + np.dot(K, np.dot(self.measurement_noise_covariance, K.T))
        
        return self.state

def cannonball_trajectory(v0, angle, g=-9.8, dt=0.1, total_time=10):
    """
    Simulates the trajectory of a cannonball.
    
    Arguments:
    - v0: Initial velocity (m/s)
    - angle: Launch angle (degrees)
    - g: Acceleration due to gravity (m/s^2)
    - dt: Time step for simulation (seconds)
    - total_time: Total simulation time (seconds)
    
    Returns:
    - x_coordinates: List of x-coordinates at each time step
    - y_coordinates: List of y-coordinates at each time step
    """
    x_coordinates = []
    y_coordinates = []
    
    theta = math.radians(angle)
    vx0 = v0 * math.cos(theta)
    vy0 = v0 * math.sin(theta)
    
    t = 0
    while t <= total_time:
        x = vx0 * t
        y = vy0 * t + 0.5 * g * t**2
        x_coordinates.append(x)
        y_coordinates.append(y)
        t += dt
    
    return x_coordinates, y_coordinates

def add_noise(value, noise_std):
    """
    Adds Gaussian noise to a given value.
    
    Arguments:
    - value: The value to add noise to
    - noise_std: Standard deviation of the Gaussian noise
    
    Returns:
    - The noisy value
    """
    noise = random.gauss(0, noise_std)
    noisy_value = value + noise
    return noisy_value

# Example usage

initial_velocity = 50.0  # m/s
launch_angle = 45.0  # degrees
launch_angle_rad = math.radians(launch_angle)
initial_x_vel = initial_velocity*math.cos(launch_angle_rad)
initial_y_vel = initial_velocity*math.sin(launch_angle_rad)

angle_noise_std = 0.01  # Standard deviation of angle noise (degrees)
range_noise_std = 1.0  # Standard deviation of range noise (m)

# Simulate cannonball trajectory
x_coordinates, y_coordinates = cannonball_trajectory(initial_velocity, launch_angle)

# Generate noisy sensor measurements
sensor_angles = [add_noise(math.atan2(y, x), angle_noise_std) for x, y in zip(x_coordinates, y_coordinates)]
sensor_ranges = [add_noise(math.sqrt(x**2 + y**2), range_noise_std) for x, y in zip(x_coordinates, y_coordinates)]

# Set up Kalman filter

initial_state = np.array([x_coordinates[0], initial_x_vel, 
                          y_coordinates[0], 
                          initial_y_vel,-9.8])
initial_covariance = 3.0*np.eye(5)
process_noise_covariance = 0.0001 * np.eye(5)  # Process noise covariance matrix
measurement_noise_covariance = np.diag([angle_noise_std**2, range_noise_std**2])  # Measurement noise covariance matrix
kf = KalmanFilter(initial_state, initial_covariance, process_noise_covariance, measurement_noise_covariance)

# Set up the plot
fig, ax = plt.subplots()


plt.xlim(-100, max(x_coordinates) + 100)
plt.ylim(0, max(y_coordinates) + 200)
plt.gca().set_aspect('equal', adjustable='box')
# Create lists to store historical data
true_trajectory = []
measured_positions = []
kalman_estimates = []
kalman_predictions = []



def frame_update(frame):
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
    angle_noisy = sensor_angles[frame]  # Noisy angle measurement
    range_noisy = sensor_ranges[frame]  # Noisy range measurement
    x_meas = range_noisy * math.cos(angle_noisy)  # Calculate x-coordinate from range and angle
    y_meas = range_noisy * math.sin(angle_noisy)  # Calculate y-coordinate from range and angle
    plt.scatter(x_meas, y_meas, color='red', label='Measured Position')

    # Kalman filter update
    measurement = np.array([x_meas, y_meas])
    H = np.array([[1,0,0,0,0],[0,0,1,0,0]])

    kf.update(measurement,H)
    kf_state = kf.state
    plt.scatter(kf_state[0], kf_state[2], color='green', label='Kalman Estimate')

    # Kalman filter prediction
    A = np.array([[1, dt, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, dt, 0],
              [0, 0, 0, 1, 0.5 *  dt**2],
              [0, 0, 0, 0, 1]])


    kf.predict(A)
    kf_state_predicted = kf.state
    if frame<10:
        print(f'****** Frame {frame} ********')
        print('Measurement:',measurement)
        print('Updated state:',kf_state)
        print('Predicted State:',kf_state_predicted)
    plt.scatter(kf_state_predicted[0], kf_state_predicted[2], color='blue', label='Kalman Prediction')

    # Append current values to the historical data lists
    true_trajectory.append((x, y))
    measured_positions.append((x_meas, y_meas))
    kalman_estimates.append((kf_state[0], kf_state[2]))
    kalman_predictions.append((kf_state_predicted[0], kf_state_predicted[2]))
    
   # Plot historical traces
    plt.plot(*zip(*true_trajectory), color='black', linestyle='-', linewidth=1, label='True Trajectory')
    plt.plot(*zip(*measured_positions), color='red', linestyle=':', linewidth=1, label='Measured Position')
    plt.plot(*zip(*kalman_estimates), color='green', linestyle='-', linewidth=1, label='Kalman Estimate')
    plt.plot(*zip(*kalman_predictions), color='blue', linestyle='--', linewidth=1, label='Kalman Prediction')



    plt.title('Cannonball Trajectory')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Distance (m)')
    plt.legend()
    plt.grid(True)
    if frame == len(x_coordinates) - 1:
        # Animation loop finished, reset traces
        true_trajectory.clear()
        measured_positions.clear()
        kalman_estimates.clear()
        kalman_predictions.clear()
        kf.__init__(initial_state, initial_covariance, process_noise_covariance, measurement_noise_covariance)


animation = FuncAnimation(fig, frame_update, frames=len(x_coordinates), interval=200)

plt.show()
