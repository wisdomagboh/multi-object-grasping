
import sys
import subprocess
import rospy
import time
import timeit
import random
import IPython

import numpy as np
import math as m

from PIL import Image
from copy import copy
from pyquaternion import Quaternion
from dm_control import suite
from dm_control import viewer
from dm_control.suite import pusher
from dm_control.utils import inverse_kinematics as ik
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from robotiq_c_model_control.msg import CModel_robot_input
from sensor_msgs.msg import JointState

from resources import *

scene_number = int(sys.argv[1])
grasp_type = str(sys.argv[2])

rw_data_path = './rw_data/'

subprocess.call(['mkdir', '-p', rw_data_path+'scene_{}'.format(scene_number)])

class mog():

    def __init__(self):

        pusher_xml_path = suite.__path__[0]+"/pusher_clutter.xml"
        subprocess.call(["sudo", "cp", "./mog_robot_objects.xml", pusher_xml_path])

        self.env = suite.load(domain_name="pusher", task_name="easy")
        self.physics = self.env.physics
        sim_qpos = self.physics.data.qpos[:]
        num_objs = self.count_objects(sim_qpos)
        self.load_constants(num_objs)

        self.physics.model.opt.integrator = self.SIM_INTEGRATOR
        self.physics.model.opt.timestep = self.SIM_TIMESTEP
        self.physics.model.cam_quat[:] = self.CAM_QUAT
        self.physics.model.cam_pos[:] = self.CAM_POS

        self.physics_ik = self.physics.copy(share_model=False)
        self.physics_real = self.physics.copy()

        # Object state
        self.mj_state_sub = rospy.Subscriber('/mj_state', Float32MultiArray, self.mj_state_callback)

        self.tcomp_sub = rospy.Subscriber('/task_complete', Float32MultiArray, self.tcomp_callback)

        # Robot state
        self.robot_state_sub = rospy.Subscriber('/joint_states', JointState, self.robot_state_callback)

        # Gripper state
        self.gripper_state_sub = rospy.Subscriber('/CModelRobotInput', CModel_robot_input, self.gripper_state_callback)

        # Arm joint publisher
        self.arm_pub = rospy.Publisher('/arm_command', Float32MultiArray, queue_size=1)
        self.arm_dur_pub = rospy.Publisher('/arm_dur_command', Float32MultiArray, queue_size=1)
        self.grip_pub = rospy.Publisher('/grip_command', Float32MultiArray, queue_size=1)
        self.rate = rospy.Rate(5)

        self.grasp_plan_pub = rospy.Publisher('/grasp_plan_cmd', String, queue_size=1)

        rospy.Subscriber('/planned_grasp', Float32MultiArray, self.pg_callback, queue_size=1)

    def pg_callback(self, data):
        return data.data

    def gripper_state_callback(self, data):

        gripper_val = data.gPR

        if gripper_val>50:
            # Gripper is closed
            self.gripper_state = [0., 0.]
        else:
            # Gripper is open
            self.gripper_state = [-0.05, 0.05]

        return None

    def robot_state_callback(self, data):
        if data.name == JOINT_NAMES:
            self.robot_state = np.array(data.position)
        return None

    def mj_state_callback(self, data):
        mj_state_list = data.data
        nobjects = int(len(mj_state_list)/3.)
        self.mj_state = np.reshape(mj_state_list, (nobjects, 3))
        return None

    def tcomp_callback(self, data):
        print ('Data received')
        return None

    def count_objects(self, qpos):
        # Doesnt matter
        return 2

    def load_constants(self, num_objs):

        # Simulator parameters
        self.SIM_TIMESTEP = 1e-3
        self.SIM_INTEGRATOR = 1
        self.CAM_QUAT = np.array([0., 0.,0.,1.])
        self.CAM_POS = np.array([1.7, 0.,1.35]) #1.7

        # Robot parameters
        self.DOF = 6 # For only the UR5
        self.N_GRIPPER_JOINTS = 4

        # Inverse kinematics
        self.SITE_NAME = "ee_point"      # Point in the end-effector for ik solutions
        self.JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.IK_TOL = 1e-14              # IK solver tolerance
        self.IK_TOL_LARGE = 1e-10        # IK solver larger tolerance
        self.MAX_STEPS = 100             # Max. number of steps for IK solver

        # Objects in the environment
        self.NUM_OBJS = num_objs
        self.N_QPOS = self.DOF + self.NUM_OBJS*7 + self.N_GRIPPER_JOINTS
        self.N_QVEL = self.DOF + self.NUM_OBJS*6 + self.N_GRIPPER_JOINTS

        # Robot params
        self.Z_ABOVE = 0.63
        self.Z_TOP = 0.478
        self.BOX_LOCATION = [1.87, 0.21]
        self.OBJECT_Z = 0.41

        self.up_time = 1
        self.travel_time = 3
        self.reach_time = 1
        self.travel_pts = 375
        self.up_pts = 125
        self.total_pregrasp_pts = self.travel_pts + self.up_pts

        # Contact parameters
        obj_names = ["o1_1", "o2_1"]
        self.object_ids = []
        self.interest_ids = []
        for obj_name in obj_names:
            obj_id = self.env.physics.named.model.geom_bodyid[obj_name]
            self.object_ids.append(obj_id)
            self.interest_ids.append(obj_id)

        # Add jaws to interest ids
        #IPython.embed()
        left_jaw_id = self.env.physics.named.model.geom_bodyid["rl_plate_geom"]
        right_jaw_id = self.env.physics.named.model.geom_bodyid["rr_plate_geom"]
        self.interest_ids.append(left_jaw_id)
        self.interest_ids.append(right_jaw_id)

        # Object joint names
        self.obj_joint_names = ['blue_pentagon_joint', 'blue_quad_joint']
        self.MIN_GRASP_FORCE = 25 # Newtons

        # Steps
        steps_per_second = int(1/self.SIM_TIMESTEP)

        self.jaw_steps = 2*steps_per_second
        self.up_steps = self.up_time*steps_per_second
        self.t_steps = self.travel_time*steps_per_second
        self.total_num_steps = 2*self.t_steps + 2*self.up_steps + 2*self.jaw_steps
        self.grasp_end_steps = self.t_steps+self.up_steps+self.jaw_steps
        self.grasp_start_steps = self.t_steps+self.up_steps
        self.obj_drop_steps = 2*self.t_steps + 2*self.up_steps + self.jaw_steps

        self.g1_pos_init = [0, 0.0425]
        self.g2_pos_init = [0, -0.0425]
        self.g1_quat_init = [1, 0, 0, 0]
        self.g2_quat_init = [1, 0, 0, 0]

    def visualize_robot_ctrls(self, x_0, j_ps, cand_grasp):
        self.reset_viewer_params(x_0, j_ps, cand_grasp)
        self.view()

    def reset_viewer_params(self, x_0, j_ps, cand_grasp):
        self.count_actions = 0
        self.init_state = x_0.copy()
        self.des_joint_positions = j_ps.copy()
        self.MAX_ACTIONS_STEPS = self.des_joint_positions.shape[0]
        self.set_state(x_0)
        self.close_jaw_steps = np.linspace(start=0, stop=0.0425, num=self.jaw_steps)

        self.g1_pos, self.g2_pos, self.g1_quat, self.g2_quat = self.get_gripper_params(cand_grasp)

        site_rot_mat_serial = self.env.physics.named.data.site_xmat['ee_point'].copy()
        site_rot_mat = site_rot_mat_serial.reshape(3,3)
        obj_quat_val = self.env.physics.named.data.qpos['blue_quad_joint'][3:]
        obj_quat = Quaternion(obj_quat_val)
        obj_rot_mat = obj_quat.rotation_matrix
        obj_site_rel_rot_mat = np.dot(np.transpose(site_rot_mat), obj_rot_mat)

    def set_state(self, x_0):
        with self.env.physics.reset_context():
            self.env.physics.set_state(x_0)
            try:
                self.env.physics.step()
            except:
                pass

        return None

    def view(self):
        viewer.launch(self.env, policy=self.policy)

    def viz_jaw_pos(self, q_l, q_r, qvel_l, qvel_r):
        self.env.physics.named.data.qpos['lp_y'] = q_l
        self.env.physics.named.data.qpos['rp_y'] = q_r
        self.env.physics.named.data.qvel['lp_y'] = qvel_l
        self.env.physics.named.data.qvel['rp_y'] = qvel_r
        self.env.physics.named.data.qacc['lp_y'] = 0
        self.env.physics.named.data.qacc['rp_y'] = 0
        return None

    def viz_arm_pos(self, jpos):
        self.env.physics.data.qpos[0:self.DOF] = jpos
        self.env.physics.data.qvel[0:self.DOF] = np.zeros(self.DOF)
        self.env.physics.data.qacc[0:self.DOF] = np.zeros(self.DOF)

    def real_jaw_pos(self, g1_quat, g2_quat, g1_pos, g2_pos):
        self.env.physics.named.model.body_quat['rl_plate'] = g1_quat
        self.env.physics.named.model.body_quat['rr_plate'] = g2_quat
        self.env.physics.named.model.body_pos['rl_plate'][0:2] = g1_pos
        self.env.physics.named.model.body_pos['rr_plate'][0:2] = g2_pos

    def policy(self, timestep):
        # Set general actuator controls
        ctrl = np.zeros(self.DOF+self.N_GRIPPER_JOINTS)

        # Set joint positions
        if self.count_actions < self.obj_drop_steps:
            jpos = self.des_joint_positions[self.count_actions].copy()
        else:
            jpos = self.des_joint_positions[-1].copy()
        self.viz_arm_pos(jpos)

        # Set default jaw positions (open)
        # Set gripper position
        self.viz_jaw_pos(0.,0.,0,0)

        # Set state
        if self.count_actions == 0:
            # Set the first state
            self.set_state(self.init_state)
            # Set real jaw positions
            self.real_jaw_pos(self.g1_quat, self.g2_quat, self.g1_pos, self.g2_pos)

        elif self.count_actions >= self.grasp_start_steps and self.count_actions < self.grasp_end_steps:
            # Set close gripper controls
            ctrl[-2:] = [-1, 1]
            self.env.physics.data.ctrl[-2:] = np.array([-1., 1.])

            # Set invisible gripper position to visible gripper position
            left_slide_pos = self.env.physics.named.data.qpos['left_slide']
            right_slide_pos = self.env.physics.named.data.qpos['right_slide']

            left_slide_vel = self.env.physics.named.data.qvel['left_slide']
            right_slide_vel = self.env.physics.named.data.qvel['right_slide']

            #print ('left slide vel', left_slide_vel)
            #print ('Right slide vel', right_slide_vel)

            self.inviz_l_jaw_pos = -right_slide_pos
            self.inviz_r_jaw_pos = left_slide_pos
            self.inviz_l_jaw_vel = -right_slide_vel
            self.inviz_r_jaw_vel = left_slide_vel
            self.viz_jaw_pos(self.inviz_l_jaw_pos, self.inviz_r_jaw_pos,
                            self.inviz_l_jaw_vel, self.inviz_r_jaw_vel)

            #self.grasped_obj_list = self.identify_grasped_objects()

        elif self.count_actions == self.grasp_end_steps:
            # Grasp ended
            self.grasped_obj_list = self.identify_grasped_objects()
            # Get relative pose of objects in gripper
            self.obj_transforms = []
            for obj in self.grasped_obj_list:
                obj_transform = self.compute_ee_obj_transform(obj_name=obj)
                self.obj_transforms.append(obj_transform)

            self.viz_jaw_pos(self.inviz_l_jaw_pos, self.inviz_r_jaw_pos,
                            self.inviz_l_jaw_vel, self.inviz_r_jaw_vel)

            self.fin_jaw = [copy(self.inviz_l_jaw_pos), copy(self.inviz_r_jaw_pos)]

            # Move real grippers away from scene
            self.real_jaw_pos(self.g1_quat_init, self.g2_quat_init,
                                self.g1_pos_init, self.g2_pos_init)

        elif self.count_actions > self.grasp_end_steps and self.count_actions < self.obj_drop_steps:
            # Use relative pose to set object positions
            # Get ee_point
            ee_point_pos = self.env.physics.named.data.site_xpos['ee_point'][0:3]
            curr_site_rot_mat_serial = self.env.physics.named.data.site_xmat['ee_point'].copy()
            curr_site_rot_mat = np.reshape(curr_site_rot_mat_serial, (3,3))

            for obj_num in range(len(self.grasped_obj_list)):
                obj_name = self.grasped_obj_list[obj_num]
                updated_obj_pos = ee_point_pos - self.obj_transforms[obj_num][0]
                self.env.physics.named.data.qpos[obj_name][0:3] = updated_obj_pos.copy()
                self.env.physics.named.data.qvel[obj_name] = np.zeros(6)
                self.env.physics.named.data.qacc[obj_name] = np.zeros(6)

                obj_site_rel_rot_mat = self.obj_transforms[obj_num][1]
                curr_obj_rot_mat = np.dot(curr_site_rot_mat, obj_site_rel_rot_mat)
                curr_obj_quat = Quaternion(matrix=curr_obj_rot_mat).elements
                self.env.physics.named.data.qpos[obj_name][3:] = curr_obj_quat.copy()

            # Also keep gripper at current position
            self.viz_jaw_pos(self.fin_jaw[0], self.fin_jaw[1], 0, 0)
        else:
            # Compute new grasp action
            pass

        self.count_actions += 1

        return ctrl

    def identify_grasped_objects(self):
        # A grasped object should be in wrench equilibrium.
        # A grasped object would be in contact with at least two other objects

        # Useful contact data
        n_con = copy(self.env.physics.data.ncon)
        geom_1_contacts = copy(self.env.physics.data.contact['geom1'])
        geom_2_contacts = copy(self.env.physics.data.contact['geom2'])

        geom_1_ids = []
        geom_2_ids = []
        for k in range(geom_1_contacts.shape[0]):
            geom_1_id = self.env.physics.model.geom_bodyid[geom_1_contacts[k]]
            geom_2_id = self.env.physics.model.geom_bodyid[geom_2_contacts[k]]
            geom_1_ids.append(geom_1_id)
            geom_2_ids.append(geom_2_id)

        contact_positions = copy(self.env.physics.data.contact['pos'])
        contact_frames = copy(self.env.physics.data.contact['frame'])
        contact_dists = copy(self.env.physics.data.contact['dist'])

        grasped_objects = []
        for obj_num in range(self.NUM_OBJS):
            object_id = copy(self.object_ids[obj_num])
            contact_id = 0
            all_contacts = []
            object_valid_contacts = []
            geoms_in_contact = []
            object_grasped = False
            for contact_id in range(n_con):
                c_geom_1 = geom_1_ids[contact_id]
                c_geom_2 = geom_2_ids[contact_id]

                # Extract contacts of interest: other objects + jaws
                # Get all contacts for current object
                if c_geom_1 == object_id or c_geom_2 == object_id:
                    # Get contacts of interest: other objects or jaws
                    if c_geom_1 == object_id and np.any(c_geom_2 == self.interest_ids):
                        contact_of_interest = True
                    elif np.any(c_geom_1 == self.interest_ids) and c_geom_2 == object_id:
                        contact_of_interest = True
                    else:
                        contact_of_interest = False

                    if contact_of_interest:
                        object_valid_contacts.append(contact_id)
                        geoms_in_contact.append([c_geom_1, c_geom_2])

            con_num = 0
            all_world_frame_forces = []
            for contact_id in object_valid_contacts:
                cforce = copy(self.env.physics.data.contact_force(contact_id))
                cdist = contact_dists[contact_id]
                cpos = contact_positions[contact_id]
                cframe = contact_frames[contact_id]
                # print ('\n')
                # print ('Contacts for object', obj_num)
                # print ('*Contact*', contact_id)
                # print ('cf:', cforce)
                # print ('cdist:', cdist)
                # print ('cpos:', cpos)
                # print ('cframe', cframe)
                # print ('Cgeoms', geoms_in_contact[con_num])
                con_num += 1

                R_cframe_wrt_W = np.reshape(cframe, (3,3))
                contact_force_in_cframe = cforce[0]
                world_frame_force = np.dot(R_cframe_wrt_W, contact_force_in_cframe)
                all_world_frame_forces.append(world_frame_force)

            # Get force norms
            all_force_vec_norms = []
            for force_vec in all_world_frame_forces:
                force_vec_norm = np.linalg.norm(force_vec)
                all_force_vec_norms.append(force_vec_norm)

            wff = np.array(all_world_frame_forces)
            net_force = np.linalg.norm(np.sum(wff, axis=0))
            print ('Object', obj_num)
            print ('Net force is', net_force)
            # We want the total force on a grasped object to be zero
            # in the world frame - due to uneven actuation it could be slightly
            # greater.
            try:
                max_force = np.max(all_force_vec_norms)
            except:
                max_force = 0

            if np.any(np.array(all_force_vec_norms) >= self.MIN_GRASP_FORCE) and net_force < max_force:
                # Assume antipodal grasp exists
                # If there are at least two forces that cancel out
                grasped_objects.append(obj_num)

        grasped_obj_names = []
        for obj_num in grasped_objects:
            grasped_obj_names.append(self.obj_joint_names[obj_num])

        return grasped_obj_names

    def compute_ee_obj_transform(self, obj_name='blue_pentagon_joint'):
        # Translations
        ee_point_pos = self.env.physics.named.data.site_xpos['ee_point'][0:3]
        obj_pt = self.env.physics.named.data.qpos[obj_name][0:3]
        print ('ee_point', ee_point_pos)
        print ('obj_pt', obj_pt)
        delta_pos = ee_point_pos - obj_pt

        # Rotations
        site_rot_mat_serial = self.env.physics.named.data.site_xmat['ee_point'].copy()
        site_rot_mat = site_rot_mat_serial.reshape(3,3)
        obj_quat_val = self.env.physics.named.data.qpos[obj_name][3:]
        obj_quat = Quaternion(obj_quat_val)
        obj_rot_mat = obj_quat.rotation_matrix
        obj_site_rel_rot_mat = np.dot(np.transpose(site_rot_mat), obj_rot_mat)

        return [delta_pos, obj_site_rel_rot_mat]

    def grasp_action(self, x_0, q_x, q_y, q_theta):
        """Generates a grasp action given x,y,theta
        0. Get current robot qpos (use only arm)
        1. Go above grasp location
        2. Downward action
        3. Close grippers
        4. Upward action
        5. Go above box location
        6. Open grippers
        """

        # Generate a robot qpos for each of the key points
        u_1 = [q_x, q_y, self.Z_ABOVE, q_theta]
        u_2 = [q_x, q_y, self.Z_TOP, q_theta]
        q_x_box = self.BOX_LOCATION[0]
        q_y_box = self.BOX_LOCATION[1]
        q_theta_box = 0.
        u_3 = [q_x_box, q_y_box, self.Z_ABOVE, q_theta_box]

        start_qpos_1 = x_0[0:self.DOF].copy()
        robot_qpos_above = self.generate_robot_ik(start_qpos_1, u_1)  # Above the x,y,theta pose
        start_qpos_2 = robot_qpos_above.copy()
        robot_qpos_down = self.generate_robot_ik(start_qpos_2, u_2)  # On the x,y,theta pose
        start_qpos_3 = robot_qpos_above.copy()
        robot_qpos_box = self.generate_robot_ik(start_qpos_3, u_3)   # On the box

        ctrl_mag = 10
        # jaw_1 = ctrl_mag*np.repeat([[1,-1]], self.t_steps, axis=0) # open
        # jaw_2 = ctrl_mag*np.repeat([[1,-1]], self.up_steps, axis=0) # open
        # jaw_3 = ctrl_mag*np.repeat([[-1, 1]], self.jaw_steps, axis=0) # close
        # #jaw_3 = np.linspace(start=[-1, 1], stop=[0, 0], num=self.jaw_steps)
        # jaw_4 = ctrl_mag*np.repeat([[-1, 1]], self.up_steps, axis=0) # close
        # jaw_5 = ctrl_mag*np.repeat([[-1, 1]], self.t_steps, axis=0) # close
        # jaw_6 = ctrl_mag*np.repeat([[1, -1]], self.jaw_steps, axis=0) # open

        jaw_1 = ctrl_mag*np.repeat([[1,-1]], self.t_steps, axis=0) # open
        jaw_2 = ctrl_mag*np.repeat([[1,-1]], self.up_steps, axis=0) # open
        jaw_3 = ctrl_mag*np.repeat([[-1, 1]], self.jaw_steps, axis=0) # close
        #jaw_3 = np.linspace(start=[-1, 1], stop=[0, 0], num=self.jaw_steps)
        jaw_4 = ctrl_mag*np.repeat([[-1, 1]], self.up_steps, axis=0) # close
        jaw_5 = ctrl_mag*np.repeat([[-1, 1]], self.t_steps, axis=0) # close
        jaw_6 = ctrl_mag*np.repeat([[1, -1]], self.jaw_steps, axis=0) # open


        robot_ctrls_1 = np.linspace(start=robot_qpos_box, stop=robot_qpos_above, num=self.t_steps)
        robot_ctrls_2 = np.linspace(start=robot_qpos_above, stop=robot_qpos_down, num=self.up_steps)
        robot_ctrls_3 = np.linspace(start=robot_qpos_down, stop=robot_qpos_down, num=self.jaw_steps)
        robot_ctrls_4 = np.linspace(start=robot_qpos_down, stop=robot_qpos_above, num=self.up_steps)
        robot_ctrls_5 = np.linspace(start=robot_qpos_above, stop=robot_qpos_box, num=self.t_steps)
        robot_ctrls_6 = np.linspace(start=robot_qpos_box, stop=robot_qpos_box, num=self.jaw_steps)

        robot_positions = np.zeros((self.total_num_steps, self.DOF))
        # jaw_positions = np.zeros((self.total_num_steps, self.N_GRIPPER_JOINTS))

        robot_positions[0:self.t_steps] = robot_ctrls_1.copy()
        robot_positions[self.t_steps:self.t_steps+self.up_steps] =robot_ctrls_2.copy()
        robot_positions[self.t_steps+self.up_steps:self.t_steps+self.up_steps+self.jaw_steps] = robot_ctrls_3.copy()
        robot_positions[self.t_steps+self.up_steps+self.jaw_steps:self.t_steps+2*self.up_steps+self.jaw_steps] = robot_ctrls_4.copy()
        robot_positions[self.t_steps+2*self.up_steps+self.jaw_steps: 2*self.t_steps+2*self.up_steps+self.jaw_steps] = robot_ctrls_5.copy()
        robot_positions[2*self.t_steps+2*self.up_steps+self.jaw_steps:2*self.t_steps+2*self.up_steps+2*self.jaw_steps] = robot_ctrls_6.copy()

        # Add more waypoints.

        pregrasp_phase_1 = np.linspace(start=start_qpos_1, stop=robot_qpos_above, num=self.travel_pts)
        pregrasp_phase_2 = np.linspace(start=robot_qpos_above, stop=robot_qpos_down, num=self.up_pts+1)
        pregrasp_jnt_arr = np.zeros((self.total_pregrasp_pts, 6))
        pregrasp_jnt_arr[0:self.travel_pts] = pregrasp_phase_1.copy()
        pregrasp_jnt_arr[self.travel_pts:self.travel_pts+self.up_pts] = pregrasp_phase_2[1:].copy()
        pregrasp_jnt_seq = pregrasp_jnt_arr.reshape(1,self.total_pregrasp_pts*6).tolist()[0]

        t_preg_0 = self.reach_time
        t_preg_1 = self.travel_time+self.reach_time
        t_preg_2 = self.up_time+self.travel_time+self.reach_time
        preg_dur_phase_1 = np.linspace(start=t_preg_0, stop=t_preg_1, num=self.travel_pts)
        preg_dur_phase_2 = np.linspace(start=t_preg_1, stop=t_preg_2, num=self.up_pts+1)
        preg_dur = np.zeros((self.total_pregrasp_pts))
        preg_dur[0:self.travel_pts] = preg_dur_phase_1.copy()
        preg_dur[self.travel_pts:self.total_pregrasp_pts] = preg_dur_phase_2[1:].copy()
        pregrasp_jnt_dur = preg_dur.tolist()

        #IPython.embed()
        #postgrasp_jnt_arr = np.array([robot_qpos_above, robot_qpos_box])
        postgrasp_jnt_arr = np.linspace(start=robot_qpos_above, stop=robot_qpos_box, num=self.travel_pts)
        postgrasp_jnt_seq = postgrasp_jnt_arr.reshape(1,self.travel_pts*6).tolist()[0]
        #postgrasp_jnt_dur = [self.up_time, self.up_time+self.travel_time]
        postgrasp_jnt_dur_arr = np.linspace(start=self.up_time, stop=self.travel_time+self.up_time, num=self.travel_pts)
        postgrasp_jnt_dur = postgrasp_jnt_dur_arr.tolist()

        return robot_positions, pregrasp_jnt_seq, postgrasp_jnt_seq, pregrasp_jnt_dur, postgrasp_jnt_dur

    def get_robot_ik(self, des_pos, des_quat=None):
        q_pos = ik.qpos_from_site_pose(physics=self.physics_ik, site_name=self.SITE_NAME,
                                target_pos=des_pos,
                                target_quat=des_quat,
                                joint_names=self.JOINTS,tol=self.IK_TOL,
                                max_steps=self.MAX_STEPS,
                                inplace=True)

        if q_pos.success != True:
            print ('solution not found')
            q_pos = ik.qpos_from_site_pose(physics=self.physics_ik, site_name=self.SITE_NAME,
                            target_pos=des_pos,
                            target_quat=des_quat,
                            joint_names=self.JOINTS,tol=self.IK_TOL_LARGE,
                            max_steps=self.MAX_STEPS,
                            inplace=True)
        return q_pos.qpos[0:self.DOF]

    def generate_robot_ik(self, start_qpos, u):
        start_state = self.physics_ik.get_state()
        start_state[0:self.DOF] = start_qpos.copy()

        self.set_state_ik(start_state)
        des_ee_pos = u[0:3].copy()
        des_theta = u[3]
        downward_axis = [0., 0., -1]
        ee_quat = Quaternion(axis=downward_axis, angle=des_theta)
        ee_quat_2 = Quaternion(axis=[0, 1, 0], angle=m.pi)
        des_ee_quat = (ee_quat*ee_quat_2).elements
        des_ee_qpos = self.get_robot_ik(des_ee_pos, des_quat=des_ee_quat)
        return des_ee_qpos.copy()

    def set_state_ik(self, x_0):

        with self.physics_ik.reset_context():
            self.physics_ik.set_state(x_0)
            try:
                self.physics_ik.step()
            except:
                pass

        return None

    def get_rw_state(self):

        x_0 = self.physics.get_state()

        print ('Waiting for message 1')
        rospy.wait_for_message('/CModelRobotInput', CModel_robot_input, timeout=1000)
        print ('Waiting for message 2')
        rospy.wait_for_message('/mj_state', Float32MultiArray, timeout=1000)
        print ('Waiting for message 3')
        joint_data = rospy.wait_for_message("/joint_states", JointState, timeout=WAIT_TIMEOUT)

        curr_gripper_pos = self.gripper_state.copy()
        curr_robot_qpos = self.robot_state.copy()
        curr_object_states = self.mj_state.copy()

        # Get robot qpos
        x_0[0:self.DOF] = curr_robot_qpos.copy()
        # Get gripper state - open or close
        x_0[self.DOF: self.DOF+2] = curr_gripper_pos.copy()

        # Get state of all objects
        count_objs = 0
        actual_objects = self.mj_state[0:self.NUM_OBJS].copy()
        for obj in actual_objects:
            obj_pos = np.zeros(3)
            obj_pos[0:2] = obj[0:2].copy()
            obj_pos[2] = self.OBJECT_Z
            obj_start_index = self.DOF+self.N_GRIPPER_JOINTS + count_objs*7
            obj_end_index = obj_start_index + 3
            x_0[obj_start_index:obj_end_index] = obj_pos.copy()
            count_objs += 1

        return x_0

    def get_gripper_params(self, grasp_cand):

        grasp_center = grasp_cand[0]
        grasp_orn = grasp_cand[1]

        gripper_l = GRIPPER_STROKE/2. + GRIPPER_WIDTH # Gripper half length
        lp_center_pt = grasp_center + gripper_l*np.array([m.cos(grasp_orn), m.sin(grasp_orn)])
        rp_center_pt = grasp_center - gripper_l*np.array([m.cos(grasp_orn), m.sin(grasp_orn)])
        lp_quat = Quaternion(axis=[0, 0, 1], angle=-(m.pi/2.-grasp_orn)).elements
        rp_quat = lp_quat.copy()

        return lp_center_pt, rp_center_pt, lp_quat, rp_quat

    def execute_action_real(self, x_0, U_arm, U_gripper):

        arm_action = x_0[3:9] + u[3:9]*CTRL_DUR

        stop_gripper_flag = simulate_obs_in_hand_main(curr_x0=x_0, set_sys_state=True)
        curr_gripper_action = 0
        if stop_gripper_flag:
            gripper_action = 0
            print ('Obstacle in hand detected... leaving gripper open')
        else:
            gripper_action = abs(u[9])

        self.send_command(arm_action, gripper_action)

        time.sleep(REAL_EXECUTION_DURATION)

        returned_state = get_real_state()

        return returned_state

    def execute_real_grasp(self, preg_jnt_ps, postg_jnt_ps, preg_jnt_dur, postg_jnt_dur):
        time.sleep(0.1)

        preg_jnt_ps_msg = Float32MultiArray(data=preg_jnt_ps)
        preg_jnt_dur_msg = Float32MultiArray(data=preg_jnt_dur)
        postg_jnt_ps_msg = Float32MultiArray(data=postg_jnt_ps)
        postg_jnt_dur_msg = Float32MultiArray(data=postg_jnt_dur)

        self.arm_pub.publish(preg_jnt_ps_msg)
        time.sleep(0.1)
        self.arm_dur_pub.publish(preg_jnt_dur_msg)
        time.sleep(0.1)
        self.arm_pub.publish(postg_jnt_ps_msg)
        time.sleep(0.1)
        self.arm_dur_pub.publish(postg_jnt_dur_msg)

        return None

    def get_planned_grasp(self, grasp_type):

        # Send "MOG" or "SOG"
        exp_data = grasp_type+','+'{}'.format(scene_number)
        self.grasp_plan_pub.publish(data=exp_data)

        # Wait for grasp
        grasp_data = rospy.wait_for_message('/planned_grasp', Float32MultiArray, timeout=1000)

        print ('Received grasp plan')
        if grasp_data.data != []:
            q_x, q_y, q_theta = list(grasp_data.data)
            return q_x, q_y, q_theta, True
        else:
            print ('No grasp found')
            return 0, 0, 0, False

    def main(self):
        '''Takes the grasping system type (MOG/SOG) as input. It calls the
            rw_grasp_planner_node with grasp type and waits for a grasp. Then,
            it generates and sends a full robot motion to excute the grasp.
            You can also visualize the planned robot motion in Mujoco.
            It also records the total execution time for a given grasp action.
        '''

        x_0 = self.get_rw_state()

        q_x, q_y, q_theta, grasp_exists = self.get_planned_grasp(grasp_type)

        if grasp_exists:
            start_exec_time = timeit.default_timer()
            j_ps, preg_jnt_ps, postg_jnt_ps, preg_jnt_dur, postg_jnt_dur = self.grasp_action(x_0, q_x, q_y, -q_theta+m.pi/2.)

            cand_grasp = [[q_x, q_y], q_theta]
            #self.visualize_robot_ctrls(x_0, j_ps, cand_grasp)

            print ('Sent arm command')
            # Execute in the real-world
            self.execute_real_grasp(preg_jnt_ps , postg_jnt_ps, preg_jnt_dur, postg_jnt_dur)
            print ('Waiting for task complete...')#
            t_comp = rospy.wait_for_message('/task_complete', Float32MultiArray, timeout=1000)
            print ('received task done')
            total_exec_time = timeit.default_timer() - start_exec_time
            print ('Execution time is', total_exec_time)
            np.save(rw_data_path+'scene_{}/exec_time_{}_{}'.format(scene_number, grasp_type, attempt_number), total_exec_time)
        else:
            print ('Experiment is done!-----------++++++++++++++++++')
        return None

if __name__ == '__main__':
    rospy.init_node('arm_motion_generator', anonymous=True)
    cl_mg = mog()
    joint_data = rospy.wait_for_message("/joint_states", JointState, timeout=WAIT_TIMEOUT)
    while joint_data.name != JOINT_NAMES:
        joint_data = rospy.wait_for_message("/joint_states", JointState, timeout=WAIT_TIMEOUT)
        rospy.spin()

    print ('Enter main loop.........')
    attempt_number = 0
    while not rospy.is_shutdown():
        print ('Starting afresh')
        cl_mg.main()
        attempt_number += 1
        cl_mg.rate.sleep()
