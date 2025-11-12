# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# 导入必要的库和模块
from legged_gym import LEGGED_GYM_ROOT_DIR  # 项目根目录路径
import os  # 操作系统接口

from legged_gym.envs import *  # 导入所有环境类
from legged_gym.utils import  get_args,  task_registry  # 导入工具函数和任务注册表
from terrain_base.config import terrain_config  # 导入地形配置

import torch  # PyTorch深度学习框架
import faulthandler  # 故障处理器，用于调试

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    """
    获取模型加载路径的函数
    
    Args:
        root: 模型文件所在的根目录
        load_run: 要加载的运行编号，-1表示加载最新的
        checkpoint: 要加载的检查点编号，-1表示加载最新的
        model_name_include: 模型文件名包含的关键词
        
    Returns:
        model: 模型文件名
        checkpoint: 检查点编号
    """
    if checkpoint==-1:
        # 如果未指定检查点，自动找到最新的模型文件
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))  # 按文件名排序
        model = models[-1]  # 选择最新的模型
        checkpoint = model.split("_")[-1].split(".")[0]  # 从文件名提取检查点编号
    return model, checkpoint

def play(args):
    """
    主要的游戏/测试函数
    加载训练好的模型并在环境中运行机器人
    
    Args:
        args: 命令行参数对象
    """
    faulthandler.enable()  # 启用故障处理器，便于调试
    
    # 获取实验ID和日志路径
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid
    # 确保日志目录存在（用于保存观测dump文件）
    try:
        os.makedirs(log_pth, exist_ok=True)
    except Exception as e:
        print(f"[PLAY][OBS] 创建日志目录失败: {e}")

    # 获取环境配置和训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 为测试覆盖一些参数
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0  # 禁用动作延迟

    # 设置测试环境参数
    env_cfg.env.num_envs = 10  # 并行环境数量
    env_cfg.env.episode_length_s = 1000  # 每个回合的最大时长（秒）
    env_cfg.commands.resampling_time = 60 # 命令重采样时间间隔
    env_cfg.rewards.is_play = True  # 标记为游戏/测试模式

    # 设置地形参数
    env_cfg.terrain.num_rows = 5  # 地形网格行数
    env_cfg.terrain.num_cols = 10  # 地形网格列数
    env_cfg.terrain.max_init_terrain_level = 2  # 最大初始地形难度等级

    # 设置地形高度范围
    env_cfg.terrain.height = [0.01, 0.02]
    
    # 设置深度相机参数
    env_cfg.depth.angle = [0, 1]
    
    # 设置噪声和域随机化参数
    env_cfg.noise.add_noise = True  # 添加噪声
    env_cfg.domain_rand.randomize_friction = True  # 随机化摩擦系数
    env_cfg.domain_rand.push_robots = False  # 不推机器人
    env_cfg.domain_rand.push_interval_s = 8  # 推机器人间隔
    env_cfg.domain_rand.randomize_base_mass = False  # 不随机化基座质量
    env_cfg.domain_rand.randomize_base_com = False  # 不随机化基座质心

    depth_latent_buffer = []  # 深度潜在特征缓冲区
    
    # 准备环境
    env: HumanoidRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()  # 获取初始观测

    # 调试：打印并保存初始观测信息（完整731维向量，取第0个并转CPU）
    try:
        print(f"[PLAY][OBS] 初始观测张量形状: {tuple(obs.shape)}")
        if hasattr(obs, 'shape') and obs.dim() == 2:
            print(f"[PLAY][OBS] 第0个环境的观测长度: {obs.shape[-1]}")
            obs0_np = obs[0].detach().cpu().numpy()
            print(f"[PLAY][OBS] 第0个环境的完整观测向量(共{obs0_np.shape[0]}维):\n{obs0_np}")
            # 保存到文件（同一文件中累积写入）
            dump_path = os.path.join(log_pth, "obs_dump.txt")
            with open(dump_path, "w") as f:
                f.write(f"# Initial observation shape: {tuple(obs.shape)}\n")
                f.write(f"# Env0 obs length: {obs0_np.shape[0]}\n")
                f.write("# === Initial Env0 Observation (731 dims) ===\n")
                f.write(" ".join([str(x) for x in obs0_np.tolist()]) + "\n")
            print(f"[PLAY][OBS] 初始观测已写入: {dump_path}")
    except Exception as e:
        print(f"[PLAY][OBS] 打印初始观测失败: {e}")

    # 加载策略模型
    train_cfg.runner.resume = True  # 设置为恢复模式
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root = log_pth, 
        env=env, 
        name=args.task, 
        args=args
    )
    
    # 获取推理策略
    policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    
    # 如果使用深度相机，获取深度编码器
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    # 初始化动作张量
    actions = torch.zeros(env.num_envs, 19, device=env.device, requires_grad=False)
    infos = {}
    
    # 获取深度信息
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    # 主循环：运行机器人
    for i in range(10*int(env.max_episode_length)):
       
        # 如果使用深度相机
        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                # 准备学生观测（去除深度信息）
                obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                obs_student[:, 6:8] = 0  # 清零深度相关观测
                
                # 使用深度编码器处理深度信息
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]  # 深度潜在特征
                yaw = depth_latent_and_yaw[:, -2:]  # 偏航角信息
                
            # 更新观测中的偏航角信息
            obs[:, 6:8] = 1.5*yaw
                
        else:
            depth_latent = None
        
        # 根据是否有深度actor选择不同的策略
        if hasattr(ppo_runner.alg, "depth_actor"):
            # 使用深度actor
            actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        else:
            # 使用普通策略
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        
        # 添加调试信息：打印动作范围
        if i % 50 == 0:  # 每50步打印一次
            print(f"[PLAY] Step {i}: 原始策略输出 [{actions.min().item():.4f}, {actions.max().item():.4f}]")
            print(f"[PLAY] Step {i}: 原始动作均值 {actions.mean().item():.4f}, 标准差 {actions.std().item():.4f}")
            
        # 执行动作，获取新的观测和奖励
        obs, _, rews, dones, infos = env.step(actions.detach())

        # 首次循环时再打印一次完整观测向量并追加写入文件（避免刷屏）
        if i == 0:
            try:
                print(f"[PLAY][OBS] Step {i} 后观测张量形状: {tuple(obs.shape)}")
                if hasattr(obs, 'shape') and obs.dim() == 2:
                    obs0_np = obs[0].detach().cpu().numpy()
                    print(f"[PLAY][OBS] Step {i} 第0个环境完整观测(共{obs0_np.shape[0]}维):\n{obs0_np}")
                    # 追加写入到同一文件
                    try:
                        dump_path = os.path.join(log_pth, "obs_dump.txt")
                        with open(dump_path, "a") as f:
                            f.write(f"# === After Step {i} Env0 Observation (731 dims) ===\n")
                            f.write(" ".join([str(x) for x in obs0_np.tolist()]) + "\n")
                        print(f"[PLAY][OBS] Step {i} 观测已追加到: {dump_path}")
                    except Exception as e2:
                        print(f"[PLAY][OBS] 追加写入观测失败: {e2}")
            except Exception as e:
                print(f"[PLAY][OBS] 打印Step {i}观测失败: {e}")
        
        # 打印环境处理后的动作
        if i % 50 == 0:  # 每50步打印一次
            processed_actions = env.actions  # 环境处理后的动作
            print(f"[PLAY] Step {i}: 环境处理后动作 [{processed_actions.min().item():.4f}, {processed_actions.max().item():.4f}]")
            print(f"[PLAY] Step {i}: 处理后均值 {processed_actions.mean().item():.4f}, 标准差 {processed_actions.std().item():.4f}")
            # 同步打印观测统计（概览，不再打印完整731维）
            try:
                print(f"[PLAY][OBS] Step {i}: 观测形状 {tuple(obs.shape)}, 范围 [{obs.min().item():.4f}, {obs.max().item():.4f}], 均值 {obs.mean().item():.4f}, 标准差 {obs.std().item():.4f}")
            except Exception:
                pass
            print("=" * 60)

if __name__ == '__main__':
    # 全局配置标志
    EXPORT_POLICY = False  # 是否导出策略
    RECORD_FRAMES = False  # 是否录制帧
    MOVE_CAMERA = False  # 是否移动相机
    
    # 获取命令行参数并运行
    args = get_args()
    play(args)
