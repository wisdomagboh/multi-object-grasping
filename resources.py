import numpy as np

potential_colors = ['r', 'g', 'b', 'c', 'r', 'g', 'b', 'c', 'r', 'g']

# Plot materials
plot_width = 0.08
obj_colors = potential_colors

# Gripper parameters
GRIPPER_WIDTH = 0.005 # half
GRIPPER_LEN = 0.01 # half
GRIPPER_STROKE = 0.085 # 85mm between plate centers

# Grasp sampling parameters
N_orns = 7
N_columns = 7
N_rows =  10
NUM_SAMPLE_SCENES = 10
num_x_obj_combinations = 20

# Simulator parameters
SIM_TIMESTEP = 1e-3         # Simulation time step
SIM_INTEGRATOR = 0          # Euler (1 for RK4)


t1_obj_list = ['blue_quad', 'yellow_quad','small_yellow_cube', 'red_triangle',
                'green_triangle_like','orange_triangle_like', 'blue_pentagon',
                'pink_pentagon', 'pink_hex', 'red_hex_large', 'red_hex_small']

t2_obj_list = ['r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'y_1', 'y_2', 'y_3',
                'gr_1', 'gr_2', 'gr_3', 'gr_4', 'bl_1']

t3_obj_list = ['rr_1', 'yy_1', 'yy_2', 'yy_3',
                'grp_1', 'grp_2', 'grp_3', 'blp_1', 'blp_2', 'blp_3']

t1_obj_sizes = [4, 4, 4, 3, 4, 4, 5, 5, 6, 6, 6]
t2_obj_sizes = 4*np.ones(len(t2_obj_list))
t3_obj_sizes = 4*np.ones(len(t3_obj_list))

# Widths
t1_obj_widths = [0.016, 0.016, 0.016, 0.021, 0.021, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018]
t2_obj_widths = 0.004*np.ones(len(t2_obj_list))
t3_obj_widths = 0.006*np.ones(len(t3_obj_list))

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
WAIT_TIMEOUT = 1000
