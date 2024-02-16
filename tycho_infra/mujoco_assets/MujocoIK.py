from tycho_env.utils import CHOPSTICK_CLOSE, CHOPSTICK_OPEN
from .TychoPhysics import TychoPhysics
import numpy as np
from scipy.spatial.transform import Rotation as R

NUM_ITERATION = 20

def get_IK_from_mujoco(sim, current_joint_position,
                      target_transformation,
                      target_open=None):
    bottom_chop_middle_point = target_transformation[0:3, 3]  # anchor_point

    sim.set_joint_positions(current_joint_position)
 
    rot = R.from_matrix(target_transformation[0:3,0:3]).as_euler("xyz")
    real_rot = [-rot[2] - np.pi, -rot[1], rot[0] + np.pi / 2]

    next_angles = sim.get_IK(bottom_chop_middle_point, R.from_euler("xyz", real_rot).as_quat(), NUM_ITERATION)
    if target_open:
        next_angles[-1] = np.clip(target_open, CHOPSTICK_CLOSE, CHOPSTICK_OPEN)

    return next_angles