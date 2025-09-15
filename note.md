# 观测整理
```python
def compute_observations(self):
    # 构建本体感受观测 (51维)
    obs_buf = torch.cat((
        self.base_ang_vel * self.obs_scales.ang_vel,    # 3维：角速度
        imu_obs,                                        # 2维：roll, pitch
        0*self.delta_yaw[:, None],                      # 1维：占位符
        self.delta_yaw[:, None],                        # 1维：目标偏航角
        self.delta_next_yaw[:, None],                   # 1维：下一个目标偏航角
        0*self.commands[:, 0:2],                        # 2维：占位符
        self.commands[:, 0:1],                          # 1维：线速度命令
        (self.env_class != 17).float()[:, None],       # 1维：环境类型
        (self.env_class == 17).float()[:, None],       # 1维：环境类型
        (self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos,  # 19维：关节位置
        self.dof_vel * self.obs_scales.dof_vel,        # 19维：关节速度
        self.action_history_buf[:, -1],                # 19维：上一步动作
        self.contact_filt.float()-0.5,                 # 2维：足部接触
    ), dim=-1)  # 总计51维
    
    # 构建特权信息
    priv_explicit = torch.cat((
        self.base_lin_vel * self.obs_scales.lin_vel,   # 3维：基座线速度
        0 * self.base_lin_vel,                         # 3维：占位符
        0 * self.base_lin_vel                          # 3维：占位符
    ), dim=-1)  # 总计9维
    
    priv_latent = torch.cat((
        self.mass_params_tensor,                       # 4维：质量参数
        self.friction_coeffs_tensor,                   # 1维：摩擦系数
        self.motor_strength[0] - 1,                    # 12维：电机强度1
        self.motor_strength[1] - 1                     # 12维：电机强度2
    ), dim=-1)  # 总计29维
    
    # 最终组合observation
    if self.cfg.terrain.measure_heights:
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)  # 132维
        self.obs_buf = torch.cat([
            obs_buf,                                    # 51维
            heights,                                    # 132维
            priv_explicit,                              # 9维  
            priv_latent,                                # 29维
            self.obs_history_buf.view(self.num_envs, -1)  # 51*10=510维
        ], dim=-1)
    # 总维度：51 + 132 + 9 + 29 + 510 = 731维
```

# 高度信息整理
```python
def _init_height_points(self): return points/height_points
    y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
    x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
    grid_x, grid_y = torch.meshgrid(x, y)
    points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
    points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
    points.shape() = [num_envs, num_height_points, 3]

measured_points_x = [0.0, 0.5, 1.0]
measured_points_y = [-0.2, 0.0, 0.3, 0.6]

points[0] = tensor([[ 0.0, -0.2, 0.0],
                    [ 0.0,  0.0, 0.0],
                    [ 0.0,  0.3, 0.0],
                    [ 0.0,  0.6, 0.0],
                    [ 0.5, -0.2, 0.0],
                    [ 0.5,  0.0, 0.0],
                    [ 0.5,  0.3, 0.0],
                    [ 0.5,  0.6, 0.0],
                    [ 1.0, -0.2, 0.0],
                    [ 1.0,  0.0, 0.0],
                    [ 1.0,  0.3, 0.0],
                    [ 1.0,  0.6, 0.0]])

# num_height_points = 132
self.measured_heights, self.measured_heights_data = self._get_heights()
    measured_heights.shape() = [num_envs, num_height_points]

# 提取前方高度采样点（机器人前方区域）
def _analyze_terrain_complexity(self):
        forward_heights = self.measured_heights[:, :self.cfg.terrain.front_points_num]  # 前方采样点
        forward_heights.shape() = [num_envs, 8]


    