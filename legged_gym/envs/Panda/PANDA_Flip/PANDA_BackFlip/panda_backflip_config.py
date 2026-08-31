from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class PandaBackFlipCfg( LeggedRobotCfg ):
    class env:
        # change the observation dim
        frame_stack = 10 #action stack
        c_frame_stack = 3 #critic 网络的堆叠帧数
        num_single_obs = 45 #这个是传感器可以获得到的信息
        num_observations = int(frame_stack * num_single_obs) # 10帧正常的观测
        single_num_privileged_obs = 48  #不平衡的观测，包含了特权信息，正常传感器获得不到的信息
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs) # 3帧特权观测
        num_actions = 12
        num_envs = 4096
        episode_length_s = 4 # episode length in seconds
        env_spacing = 3.  # not used with heightfields/trimeshes 
        joint_num = 12
        send_timeouts=True

        reset_height = 0.1 # [m]
        test = False


    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False# select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0., 0., 1.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
    class commands:
        curriculum = False
        max_curriculum = 0.8
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.49] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.52,
            "FL_calf_joint": -1.05,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.52,
            "FR_calf_joint": -1.05,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.52,
            "RL_calf_joint": -1.05,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.52,
            "RR_calf_joint": -1.05,
        }
        lie_joint_angles = { # = target angles [rad] when action = 0.0
            # 'FL_hip_joint': 0.0,
            # 'RL_hip_joint': 0.0,
            # 'FR_hip_joint': 0.0,
            # 'RR_hip_joint': 0.0,
            # 'FL_thigh_joint': 0.8727, #51
            # 'RL_thigh_joint': 0.9, #61
            # 'FR_thigh_joint': 0.8727, #51
            # 'RR_thigh_joint': 0.9, #61
            # 'FL_calf_joint': -1.6057, #-92
            # 'RL_calf_joint': -1.6930, #-97
            # 'FR_calf_joint': -1.6057, #-92
            # 'RR_calf_joint': -1.6930, #-97
            # 前腿：约 50° / -92°
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8727,
            "FL_calf_joint": -1.6057,

            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8727,
            "FR_calf_joint": -1.6057,

            # 后腿：约 61° / -97°
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 1.0647,
            "RL_calf_joint": -1.6930,

            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 1.0647,
            "RR_calf_joint": -1.6930,

        }
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
            "_hip_joint": 30.0,
            "_thigh_joint": 40.0,
            "_calf_joint": 65.0,
        }

        damping = {
            "_hip_joint": 0.7,
            "_thigh_joint": 1.0,
            "_calf_joint": 1.3,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda/urdf/panda3v2.urdf'
        name = "panda"
        foot_name = "foot_Link"
        penalize_contacts_on = ["thigh_Link", "calf_Link"]
        terminate_after_contacts_on = ["base_link"]
        disable_gravity = False
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
    class domain_rand:
        randomize_friction = True
        friction_range = [0.3,1.0]

        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_base_mass = True
        added_base_mass_range = [-1,1]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]  
        damping_multiplier_range = [0.9, 1.1]    


        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        add_obs_latency = True
        randomize_obs_motor_latency = True
        randomize_obs_imu_latency = True
        range_obs_motor_latency = [1, 3]
        range_obs_imu_latency = [1, 3]
        
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [1, 3]
        # Bootstrap the difficult aerial phase, then the inherited curriculum
        # removes this assistance automatically during the first ~400 updates.
        push_towards_goal = True
    class rewards:
        class scales:
            # Keep the Go2 backflip reward structure.  Panda-specific flip
            # state is retained for metrics, not used as an extra objective.
            before_setting=5.0
            line_z=25.
            angle_y=10.
            base_height_flight=5.0
            base_height_stance=10.0
            orientation=10.
            orientation_before=2.
            dof_pos=-0.2
            line_vel_stance=-1.
            ang_vel_xy=-0.2
            torques=-0.0001
            dof_pos_limits=-10.
            dof_vel_limits=-2.
            dof_vel=-0.001
            termination=0.0
            collision=-10.
            action_rate=-0.01
            feet_contact_forces=-0.1
            land_pos=1.0
            symmetric_joints=-0.3
            default_hip_pos=-0.5

        # Panda is about 2.37 times as heavy as Go2 (150 N threshold).
        max_contact_force=350
        only_positive_rewards=False
        reward_sigma=0.25
        target_height=0.72
        stance_height_target=0.37

        # One positive body-Y revolution followed by a stable feet landing.
        target_flip_angle=6.283185307179586
        min_flip_angle=5.3
        max_flip_angle=7.3
        target_ang_vel_y=7.0
        ang_vel_y_sigma=4.0
        brake_start_angle=5.0
        landing_upright_gravity_z=-0.8
        landing_max_ang_vel=3.0
        landing_min_height=0.35
        landing_settle_steps=25
        soft_dof_pos_limit=0.9
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            quat = 1.
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            quat = 0.1
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class PandaBackFlipCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        # Close to Go2's exploration, reduced slightly for Panda's stronger
        # actuators.
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-5
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        sym_loss = False
        obs_permutation = [-0.0001, -1, 2, -3, 4,
                           -5,6,-7,-8,9,-10,
                       -14,15,16,-11,12,13,-20,21,22,-17,18,19,
                       -26,27,28,-23,24,25,-32,33,34,-29,30,31,
                       -38,39,40,-35,36,37,-44,45,46,-41,42,43
                       ]
        ##command x y height
        act_permutation = [ -3, 4, 5, -0.0001, 1, 2, -9, 10, 11,-6, 7, 8,]#关节电机的对陈关系
        frame_stack = 10
        sym_coef = 1.0
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations =50000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'panda_backflip'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
