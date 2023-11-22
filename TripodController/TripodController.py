import numpy as np

class TripodController:
    def __init__(self):
        self.phase = 0
        self.step_duration = 0.5  # Duration of each step
        self.lift_height = 0.05  # Lift height of the legs
        self.body_height = 0.15
        self.crab_angle = 0.0

    def joint_angles(self,contact, t):
        leg_angles = []
        for leg in range(6):
            # Calculate the phase for the current leg
            leg_phase = (self.phase + leg * 2) % 6

            # Determine if the leg is in the stance or swing phase
            if leg_phase < 3:
                # Stance phase
                leg_angle = 0
            else:
                # Swing phase
                swing_phase = (leg_phase - 3) / 3
                leg_angle = swing_phase * np.pi

            leg_angles.extend([leg_angle, -leg_angle, -leg_angle])

        # Update the phase for the next step
        self.phase += 1
        if self.phase >= 6:
            self.phase = 0

        return np.array(leg_angles)

    def IMU_feedback(self, measured_attitude):
            return