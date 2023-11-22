import time
import numpy as np
from TripodController import TripodController

# Create an instance of the TripodController
controller = TripodController()

# Simulation parameters
simulation_duration = 5.0  # Duration of the simulation in seconds
dt = 0.5  # Time step in seconds

# Simulation loop
t = 0.0
while t < simulation_duration:
    # Get the joint angles from the controller
    joint_angles = controller.get_joint_angles(t)

    # Print the joint angles for visualization (you can replace this with your own code)
    print("Joint Angles:", joint_angles)

    # Update the simulation time
    t += dt

    # Wait for the specified time step
    time.sleep(dt)



