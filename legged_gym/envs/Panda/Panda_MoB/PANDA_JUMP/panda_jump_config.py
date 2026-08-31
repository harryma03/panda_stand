from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class PandaJumpCfg( LeggedRobotCfg ):
    class env:
        # change the observation dim
        frame_stack = 1 #action stack
        c_frame_stack = 3 #critic 网络的堆叠帧数
        num_single_obs = 47 #这个是传感器可以获得到的信息
        num_observations = int(frame_stack * num_single_obs) # 10帧正常的观测
        single_num_privileged_obs = 70  #不平衡的观测，包含了特权信息，正常传感器获得不到的信息
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs) # 3帧特权观测
        num_actions = 12
        num_envs = 4096
        episode_length_s = 3.0 # one preload -> forward jump -> landing sequence
        env_spacing = 3.  # not used with heightfields/trimeshes 
        joint_num = 12
        send_timeouts=True
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
        max_curriculum = 2.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # PandaJump keeps a fixed forward command
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [1.6, 1.6]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.45] # x,y,z [m]
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
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda/urdf/panda3v2.urdf'
        name = "panda"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
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
        randomize_friction = False
        friction_range = [0.4,0.8]

        push_robots = False
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_base_mass = False
        added_base_mass_range = [-1,1]

        randomize_link_mass = False
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = False
        added_base_com_range = [-0.02, 0.02]

        randomize_pd_gains = False
        stiffness_multiplier_range = [0.9, 1.1]  
        damping_multiplier_range = [0.9, 1.1]    


        randomize_motor_zero_offset = False
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

   # range to contain the real joint armature 

        add_obs_latency = False # enable after a nominal jump has been learned
        randomize_obs_motor_latency = False
        randomize_obs_imu_latency = False
        range_obs_motor_latency = [1, 3]
        range_obs_imu_latency = [1, 3]
        
        add_cmd_action_latency = False
        randomize_cmd_action_latency = False
        range_cmd_action_latency = [1, 3]

        # Bootstrap a brand-new policy.  The impulse is training-only and
        # fades to zero over jump_assist_decay_steps; play.py sets
        # env.test=True.  The roll assist tops the spin up at the first
        # airborne step, keeping the policy's own flip direction.
        jump_start_assistance = True
        jump_assist_vertical_velocity = 3.50
        jump_assist_forward_velocity = 1.20
        jump_assist_clearance = 0.12
        jump_assist_roll_velocity = 10.0
        # Assistance is guaranteed for the first full_steps, then decays
        # linearly to zero by decay_steps.  A from-scratch policy needs
        # thousands of iterations of demonstrated launches before it can
        # self-launch; the old 12000-step budget died inside the random
        # exploration phase and the policy settled for crouch-only.
        jump_assist_full_steps = 96000   # 4000 iters at 24 steps/iter
        jump_assist_decay_steps = 240000 # gone by 10000 iters

    class rewards:
        class scales:
            # _prepare_reward_function multiplies every scale by dt (0.02),
            # so -100 gives the intended -2.0 per terminated episode.
            termination = -100.0
            # ---- global regulators --------------------------------------
            tracking_ang_vel = 0.5    # no yaw drift: keep the flip axis clean
            torques = -0.0002
            base_height = 2.0
            collision = -1.
            action_rate = -0.002
            feet_contact_forces = -0.002
            # ---- takeoff: crouch, extend, spin up the roll ---------------
            preload_pose = 6.0
            preload_height = 4.0
            # Strong enough to force the leg extension that launches the
            # jump; kept below the old 6.0 only to free the hip asymmetry.
            takeoff_pose = 5.0
            jump_height = 14.0
            takeoff_velocity = 9.0  # vertical takeoff velocity shaping
            forward_takeoff = 9.0   # forward + vertical takeoff progress
            takeoff_roll_rate = 8.0
            takeoff_pitch_rate = -3.0
            real_air_time = 10.0
            # ---- flight: symmetric tuck + roll progress, pitch locked out
            # Boosted: the tuck is what makes the flip completable once the
            # roll assist decays, so make it the dominant flight reward.
            flight_leg_pose = 16.0
            flight_pitch = -5.0
            side_flip_progress = 20.0
            # ---- landing: level four-foot touchdown, absorb, stay up -----
            landing_position = 8.0
            landing_pose = 16.0
            touchdown_state = 8.0
            landing_foot_level = 20.0
            simultaneous_touchdown = 15.0
            landing_impact = -30.0
            landing_absorption_pose = 12.0
            landing_stability = 10.0
            # Dedicated gradient for active air righting: penalize residual
            # roll rate while descending toward touchdown.
            landing_spin = -6.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        base_height_target = 0.45
        crouch_height_target = 0.27
        crouch_thigh_target = 1.15
        crouch_calf_target = -2.10
        takeoff_thigh_target = 0.10
        takeoff_calf_target = -0.25
        # Higher jump buys air time for the full 360 deg roll.  3.5 m/s
        # gives ~0.71 s of flight: flip completes ~0.62, extension brake
        # ~0.62-0.71, touchdown right after 2*pi instead of knife-edge.
        jump_height_target = 0.55
        takeoff_velocity_target = 3.5
        forward_velocity_target = 1.55
        landing_distance_target = 0.68
        target_air_time = 0.65
        minimum_flight_clearance = 0.025
        successful_jump_clearance = 0.22
        successful_jump_air_time = 0.45
        # Side flip: roll rate at takeoff and total roll for completion.
        # Slightly reduced: the deep tuck amplifies the spin mid-air, and a
        # lower takeoff spin means less residual spin to kill at touchdown.
        takeoff_roll_rate_target = 9.0
        side_flip_roll_target = 6.283  # 2*pi
        # All four feet at the same height on descent (0 = level).
        landing_front_rear_offset = 0.0
        # Side flip: symmetric deep tuck in flight.  Feet close to the body
        # minimize the roll inertia so the flip completes earlier, leaving
        # more air time for the extension brake and the landing.  Calf stays
        # above the -2.44 joint limit.
        flight_front_thigh = 1.30
        flight_front_calf = -2.25
        flight_rear_thigh = 1.30
        flight_rear_calf = -2.25
        # Symmetric four-leg reach at touchdown.  More extended than before:
        # longer legs raise the roll inertia and brake the spin harder.
        # The absorb pose (below) takes over after contact.
        landing_front_thigh = 0.35
        landing_front_calf = -0.9
        landing_rear_thigh = 0.35
        landing_rear_calf = -0.9
        front_support_thigh = 0.90
        front_support_calf = -1.65
        rear_follow_thigh = 0.38
        rear_follow_calf = -0.82
        landing_absorb_thigh = 0.90
        landing_absorb_calf = -1.65
        landing_absorb_height = 0.32
        # The landing pose blend starts at 0.60, AFTER the flip has
        # completed (~0.60-0.62 with the tighter tuck).  The old 0.55 start
        # preempted the tuck: the flight target was mostly the landing pose,
        # so the policy never learned to tuck at all.
        landing_pose_start_phase = 0.60
        landing_pose_full_phase = 0.68
        # Phase is only a fallback.  Actual leg extension follows descending
        # base height, because touchdown timing changes as the policy improves.
        # The extension starts at ~0.92 m, i.e. after the apex AND after the
        # flip has completed, so it can brake the spin without preempting the
        # tuck (the old 1.00 m start ate into the tuck window).
        landing_extension_start_height = 0.92
        landing_extension_full_height = 0.66
        touchdown_forward_velocity_target = 0.60
        touchdown_vertical_velocity_target = -0.65
        landing_impact_force_threshold = 220.0
        preload_start_phase = 0.08
        preload_end_phase = 0.38
        takeoff_start_phase = 0.36
        flight_start_phase = 0.47
        flight_end_phase = 0.68
        landing_start_phase = 0.58
        landing_absorption_end_phase = 0.74
        landing_recovery_end_phase = 0.82
        # The tuck starts exactly at liftoff (flight_start_phase): the
        # push-off leg extension owns the takeoff window, the tuck owns
        # the flight.  Overlapping the two produced a half-extended,
        # half-tucked compromise pose.
        flight_tuck_start_phase = 0.47
        flight_tuck_end_phase = 0.64
        # Phase where the symmetric flight tuck is fully applied.
        flight_compactness_start_phase = 0.50
        max_contact_force = 600. # Panda weighs about 35.55 kg; allow jump impulses
        cycle_time=3.0
        # Resting on the back spreads the load over many contact points;
        # 150 N per point still never fired for a belly-up robot.  80 N with
        # the 5-step sustain ends belly-up episodes quickly.
        base_contact_force_threshold = 80.0
        base_contact_sustain_steps = 5
        robot_mass = 35.553
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
        add_noise = False # enable after the nominal jump is learned
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


class PandaJumpCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        sym_loss = True
        obs_permutation = [-0.0001, -1, 2, -3, -4,
                           -5,6,-7,-8,9,-10,
                       -14,15,16,-11,12,13,-20,21,22,-17,18,19,
                       -26,27,28,-23,24,25,-32,33,34,-29,30,31,
                       -38,39,40,-35,36,37,-44,45,46,-41,42,43]

        act_permutation = [ -3, 4, 5, -0.0001, 1, 2, -9, 10, 11,-6, 7, 8,]#关节电机的对陈关系
        frame_stack = 10
        sym_coef = 1.0
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 15000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'panda_jump'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = '/home/ub/panda/My_unitree_go2_gym/logs/panda_jump/Aug17_20-26-24_' # updated from load_run and chkpt
        # A converged v5 checkpoint has std ~= 0.08 and cannot discover a new
        # touchdown strategy.  Keep the actor weights but restore moderate
        # exploration only while resuming training (not during play).
        reset_noise_std_on_resume = True
        resume_noise_std = 0.12
