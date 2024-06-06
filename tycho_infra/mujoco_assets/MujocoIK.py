from tycho_env.utils import CHOPSTICK_CLOSE, CHOPSTICK_OPEN
from .TychoPhysics import TychoPhysics
import numpy as np
from scipy.spatial.transform import Rotation as R

NUM_ITERATION = 20

# Specified either as 8D vector containing xyz + quat + open
# or as 3 by 4 matrix transformation and an opening angle
def get_IK_from_mujoco(sim, current_joint_position,
                      target_transformation=None,
                      target_vector=None,
                      target_open=None):
    if target_vector is not None:
        bottom_chop_middle_point = target_vector[0:3]
        rot = R.from_quat(target_vector[3:7]).as_euler("xyz")
    else:
        bottom_chop_middle_point = target_transformation[0:3, 3]  # anchor_point
        rot = R.from_matrix(target_transformation[0:3,0:3]).as_euler("xyz")

    sim.set_joint_positions(current_joint_position)
    real_rot = [-rot[2] - np.pi, -rot[1], rot[0] + np.pi / 2]

    next_angles = sim.get_IK(bottom_chop_middle_point, R.from_euler("xyz", real_rot).as_quat(), NUM_ITERATION)
    if target_open is not None:
        next_angles[-1] = np.clip(target_open, CHOPSTICK_CLOSE, CHOPSTICK_OPEN)
    elif target_vector is not None:
        next_angles[-1] = np.clip(target_vector[-1], CHOPSTICK_CLOSE, CHOPSTICK_OPEN)
    return next_angles