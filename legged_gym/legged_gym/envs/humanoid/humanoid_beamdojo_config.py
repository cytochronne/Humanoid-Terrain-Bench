# BEAMDOJO Humanoid Robot Configuration
# 基于BEAMDOJO论文的人形机器人配置示例
# 展示双Critic网络、两阶段训练和Foothold奖励的完整配置

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HumanoidBEAMDOJOCfg(LeggedRobotCfg):
    """
    BEAMDOJO人形机器人环境配置
    实现双Critic网络和两阶段训练的完整配置
    """
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.80]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,
           'left_hip_roll_joint' : 0,
           'left_hip_pitch_joint' : -0.1,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_hip_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.1,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint' : 0,
        }

    class env(LeggedRobotCfg.env):
        """环境配置"""
        num_envs = 512
        episode_length_s = 40
        
        # 观测配置（根据实际humanoid机器人调整）
        n_scan = 225
        n_priv = 3 + 3 + 3  # 位置(3) + 速度(3) + 其他(3)
        n_priv_latent = 4 + 1 + 12 + 12  # 潜在状态维度
        n_proprio = 51  # 实际obs_buf维度：3+2+1+1+1+2+1+1+1+12+12+12+2=51
        history_len = 10
        
        # 重新计算总观测维度
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        num_actions = 12  # 12个关节动作
        
        # 启用接触信息
        include_foot_contacts = True
        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4    
    
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True            # 随机化摩擦系数
        friction_range = [0.8, 0.8]         # 恢复原始摩擦系数
        randomize_base_mass = True          # 随机化质量
        added_mass_range = [-2.0, 2.0]      # 负载质量 U(-2.0, 2.0) kg
        randomize_base_com = True           # 随机化质心位置
        added_com_range = [-0.05, 0.05]     # 质心偏移 U(-0.05, 0.05) m
        push_robots = True                   # 启用外部推力（抗干扰训练）
        push_interval_s = 8                  # 推力间隔：每8秒推一次
        max_push_vel_xy = 0.5                # 最大推力速度：±0.5 m/s

        randomize_motor = True              # 随机化电机特性
        motor_strength_range = [0.9, 1.1]   # 电机强度噪声 U(0.9, 1.1)
        
        randomize_actuator_offset = True    # 随机化执行器零位偏移
        actuator_offset_range = [-0.05, 0.05]  # 执行器偏移 U(-0.05, 0.05) rad
        
        randomize_pd_gains = True           # 随机化PD增益
        pd_gain_range = [0.85, 1.15]        # Kp/Kd噪声因子 U(0.85, 1.15)

        # 动作延迟相关参数（BeamDojo域随机化）
        delay_update_global_steps = 24 * 8000  # 延迟更新的全局步数
        action_delay = True                  # 启用动作延迟（抗干扰训练）
        action_curr_step = [1, 1, 2]        # 课程学习：保守渐进（20ms→20ms→40ms）
        action_curr_step_scratch = [0, 1, 1] # 从头训练时的延迟步数（0ms→20ms→20ms）
        action_delay_view = 1               # 动作延迟视图
        action_buf_len = 8                  # 动作缓冲区长度

    
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_with_hand.urdf'
        name = "g1_fix_upper"
        foot_name = "ankle_roll"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False    
    class commands( LeggedRobotCfg.commands ):
        """运动命令配置"""
        resampling_time = 1.0         # 命令重采样时间间隔（秒）
        heading_command = True         # 启用朝向命令模式
        ang_vel_clip = 0.1            # 角速度命令死区阈值
        lin_vel_clip = 0.1            # 线速度命令死区阈值
        
        # 策略1：智能速度生成配置
        height_adaptive_speed = False   # 启用基于高度的自适应速度
        speed_complexity_weight = 0.4  # 地形复杂度权重
        speed_gradient_weight = 0.4   # 高度梯度权重  
        speed_roughness_weight = 0.2  # 地形粗糙度权重
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.1, 1.0] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [-1.2, 1.2]
            
    class rewards(LeggedRobotCfg.rewards):
        """BEAMDOJO奖励配置"""
        class scales:
            """奖励权重配置"""
            # 密集奖励（用于第一个Critic）
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            orientation = -2.0
            base_height = -10.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            action_rate = -0.01
            smoothness = -1e-3
            stand_still = -0.05
            dof_vel = -1e-4
            dof_acc = -2.5e-8
            dof_pos_limits = -5.0
            dof_vel_limits = -1e-3
            dof_power = -2e-5
            feet_ground_parallel = -0.02
            feet_distance = 0.5
            feet_air_time = 1.0
            feet_clearance = -1.0
            # 稀疏奖励（用于第二个Critic）
            foothold = 1.0  # BEAMDOJO Foothold奖励（优化版）
            

        # BEAMDOJO双Critic奖励分离
        dense_rewards = [
            'tracking_lin_vel', 'tracking_ang_vel', 'orientation', 'base_height',
            'lin_vel_z', 'ang_vel_xy', 'torques', 'action_rate', 'dof_vel', 
            'dof_acc', 'feet_air_time', 'collision', 'feet_stumble', 'stand_still'
        ]
        sparse_rewards = [
            'foothold'
        ]
        
        # Foothold奖励配置 - 超级优化版（16-48倍性能提升）
        class foothold:
            height_tolerance = 0.05    # 高度容忍度 [m] - 唯一必需参数
            
        # 超级性能优化参数
        foothold_contact_threshold = 0.5    # 接触力阈值（降低以减少计算）
        foothold_frame_skip = 4             # 帧跳跃：每4帧计算一次（核心优化）
        foothold_foot_length = 0.12         # 脚长度 [m] 
        foothold_foot_width = 0.08          # 脚宽度 [m]
            
        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.
        
        # 覆盖基础配置中的参数，设置为论文中的值
        min_dist = 0.18                   # d_min: Minimum allowable distance between two feet (论文值)
        max_dist = 0.25                   # 最大脚间距，用于计算目标距离
        is_play = False                   # 是否为播放模式
        
    class normalization:
        """归一化配置"""
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        """噪声配置"""
        add_noise = True
        noise_level = 1.0
        
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05

    class sim:
        """仿真配置"""
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1
        
        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class HumanoidBEAMDOJOCfgPPO(LeggedRobotCfgPPO):
    """
    BEAMDOJO人形机器人PPO训练配置
    包含双Critic网络和两阶段训练的完整配置
    """
    
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class policy(LeggedRobotCfgPPO.policy):
        """策略网络配置"""
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        
        # 扫描编码器配置
        scan_encoder_dims = [128, 64, 32]
        priv_encoder_dims = [64, 20]
        tanh_encoder_output = False  # 编码器输出是否使用tanh激活
        
        # 支持双Critic的编码器
        use_double_critic = True  # 在这里可以启用双Critic
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        """BEAMDOJO PPO算法配置"""
        # 基础PPO参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
        # BEAMDOJO双Critic配置
        use_double_critic = True      # 设置为True启用双Critic
        dense_value_loss_coef = 1.0    # 密集奖励价值损失系数
        sparse_value_loss_coef = 1.0   # 稀疏奖励价值损失系数
        advantage_merge_weight = 0.5   # 优势函数合并权重
        
    class runner(LeggedRobotCfgPPO.runner):
        """训练运行器配置"""
        policy_class_name = 'ActorCriticRMADoubleReward'  # 使用 'ActorCriticRMADoubleReward' 启用双Critic
        algorithm_class_name = 'PPODoubleReward'       # 使用 'PPODoubleReward' 启用双Critic算法
        num_steps_per_env = 24
        max_iterations = 100000
        
        save_interval = 100
        experiment_name = 'humanoid_beamdojo'
        run_name = ''
        
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    class estimator(LeggedRobotCfgPPO.estimator):
        """状态估计器配置"""
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = HumanoidBEAMDOJOCfg.env.n_priv
        num_prop = HumanoidBEAMDOJOCfg.env.n_proprio
        num_scan = HumanoidBEAMDOJOCfg.env.n_scan

    class depth_encoder(LeggedRobotCfgPPO.depth_encoder):
        """深度编码器配置"""
        pass  # 使用基类配置

    # BEAMDOJO两阶段训练配置
    class training:
        """两阶段训练配置"""
        enable_two_stage = False  # 设置为True启用两阶段训练
        
        class stage1:
            """Stage1软约束训练配置"""
            min_steps = 1000000            # 最小训练步数
            max_steps = 5000000            # 最大训练步数  
            success_threshold = 0.8        # 成功率阈值
            terrain_type = "flat_with_target_perception"
            use_soft_termination = True    # 软终止：踩空不终止episode
            use_target_perception = True   # 使用目标地形感知
            
            # Stage1命令范围（全方向）
            class command_ranges:
                lin_vel_x = [-1.0, 1.0]
                lin_vel_y = [-1.0, 1.0]
                ang_vel_yaw = [-1.0, 1.0]
        
        class stage2:
            """Stage2硬约束训练配置"""
            terrain_type = "sparse_terrain"
            use_soft_termination = False   # 硬终止：踩空立即终止
            use_target_perception = False  # 不使用目标地形感知
            
            # Stage2命令范围（仅前进）
            class command_ranges:
                lin_vel_x = [-1.0, 1.0]
                lin_vel_y = [0.0, 0.0]     # 固定为0
                ang_vel_yaw = [0.0, 0.0]   # 固定为0
