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

# 导入必要的库
from copy import deepcopy  # 深拷贝功能
import os  # 操作系统接口
from datetime import datetime  # 日期时间处理
from typing import Tuple  # 类型提示
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库

# 导入强化学习相关模块
from rsl_rl.env import VecEnv  # 向量化环境基类
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runners import MultiTeacherDistillationRunner  # 添加多教师蒸馏运行器

# 导入项目相关模块
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR  # 项目路径常量
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params  # 辅助函数
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO  # 配置基类
from terrain_base.config import terrain_config  # 地形配置

class TaskRegistry():
    """
    任务注册器类
    用于管理不同机器人任务的注册、配置和创建
    支持环境创建和算法运行器的统一管理
    """
    
    def __init__(self):
        """初始化任务注册器，创建存储字典"""
        self.task_classes = {}  # 存储任务类（环境类）
        self.env_cfgs = {}      # 存储环境配置
        self.train_cfgs = {}    # 存储训练配置
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        """
        注册新任务
        
        Args:
            name: 任务名称（如'h1_2_fix'）
            task_class: 环境类
            env_cfg: 环境配置对象
            train_cfg: 训练配置对象
        """
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        """
        获取指定任务的环境类
        
        Args:
            name: 任务名称
            
        Returns:
            对应的环境类
        """
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        """
        获取指定任务的配置对象
        
        Args:
            name: 任务名称
            
        Returns:
            环境配置和训练配置的元组
        """
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # 同步随机种子
        env_cfg.seed = train_cfg.seed

        # 设置地形配置
        env_cfg.terrain = terrain_config

        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """
        创建环境实例
        
        Args:
            name: 注册的环境名称
            args: Isaac Gym命令行参数，如果为None则调用get_args()
            env_cfg: 环境配置文件，用于覆盖注册的配置
            
        Raises:
            ValueError: 如果没有找到对应名称的注册环境
            
        Returns:
            创建的环境实例和对应的配置文件
        """
        # 如果没有传入参数，获取命令行参数
        if args is None:
            args = get_args()
            
        # 检查是否有对应名称的注册环境
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
            
        if env_cfg is None:
            # 加载配置文件
            env_cfg, _ = self.get_cfgs(name)
        
        # 根据命令行参数覆盖配置（如果指定）
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)  # 设置随机种子
        
        # 解析仿真参数（先转换为字典）
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)

        # print("test=",env_cfg.terrain.num_goals)  # 调试信息

        # 创建环境实例
        env = task_class(   cfg=env_cfg,                    # 环境配置
                            sim_params=sim_params,          # 仿真参数
                            physics_engine=args.physics_engine,  # 物理引擎
                            sim_device=args.sim_device,    # 仿真设备
                            headless=args.headless,        # 无头模式
                            save=args.save)                # 保存数据
        # print("test=",env_cfg)  # 调试信息
        # print('env:', env)
        # print('env_cfg:', env_cfg)
        return env, env_cfg

    def make_alg_runner(self, log_root, env, name, args=None, init_wandb=True, **kwargs):
        """ Creates the training algorithm and runner. 

        Args:
            env (VecEnv): vectorized environment.
            name (str): experiment name.
            args: command line arguments.
            
        Returns:
            MultiTeacherDistillationRunner or OnPolicyRunner: training algorithm.
        """

        env_cfg, train_cfg = self.get_cfgs(name)  # 获取两个配置对象
        train_cfg_dict = class_to_dict(train_cfg)
        print("[TaskRegistry] Train config dict:", train_cfg_dict)
        # create experiment name
        experiment_name = train_cfg_dict["runner"]["experiment_name"]
        
        # set seed if provided as command line argument
        if args is not None and hasattr(args, 'seed') and args.seed is not None:
            train_cfg_dict["runner"]["seed"] = args.seed

        if args is not None:
            experiment_name += f"_{args.exptid}" if hasattr(args, 'exptid') and args.exptid is not None else ""
            experiment_name += f"_{args.run_name}" if hasattr(args, 'run_name') and args.run_name is not None else ""

        # create log directory
        run_dir_time = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = os.path.join(log_root, experiment_name, run_dir_time)
        os.makedirs(log_dir, exist_ok=True)
        
        # initialize wandb if requested
        if init_wandb and (args is None or not hasattr(args, 'debug') or not args.debug):
            self._init_wandb(train_cfg_dict, log_dir, args)
            train_cfg_dict["runner"]["logger"] = "wandb"
        else:
            train_cfg_dict["runner"]["logger"] = "tensorboard"
        
        # 检查是否使用多教师蒸馏训练器
        runner_class_name = train_cfg_dict.get("runner", {}).get("class_name", "OnPolicyRunner")
        
        print(f"[TaskRegistry] Using runner: {runner_class_name}")
        print(f"[TaskRegistry] Train config runner: {train_cfg_dict.get('runner', {})}")
        
        if runner_class_name == "MultiTeacherDistillationRunner":
            # 多教师蒸馏训练器
            from rsl_rl.runners import MultiTeacherDistillationRunner
            runner = MultiTeacherDistillationRunner(
                env=env,
                train_cfg=train_cfg_dict,
                log_dir=log_dir,
                device=args.rl_device if args is not None else "cuda:0",
                init_wandb=init_wandb,
                **kwargs
            )
        else:
            # 默认的PPO训练器
            runner = OnPolicyRunner(
                env=env,
                train_cfg=train_cfg_dict,
                log_dir=log_dir,
                device=args.rl_device if args is not None else "cuda:0",
                **kwargs
            )

        return runner, train_cfg
    
    def _init_wandb(self, train_cfg_dict, log_dir, args):
        """初始化wandb配置"""
        try:
            import wandb
            
            # 检查是否是多教师蒸馏训练
            is_multi_teacher = train_cfg_dict.get("runner", {}).get("class_name") == "MultiTeacherDistillationRunner"
            
            # 设置基础wandb配置
            wandb_config = {
                "project": getattr(args, 'proj_name', 'legged_gym'),
                "name": getattr(args, 'exptid', 'experiment'),
                "dir": log_dir,
                "config": train_cfg_dict,
                "save_code": True,
                "mode": "online",
                "settings": wandb.Settings(_disable_stats=True)
            }
            
            # 如果是多教师蒸馏，添加特殊标签和配置
            if is_multi_teacher:
                # 添加多教师蒸馏特定的标签
                teacher_tags = ["multi_teacher", "distillation", "h1_robot"]
                
                # 从训练配置中提取教师路径信息
                if "policy" in train_cfg_dict and "teacher_model_paths" in train_cfg_dict["policy"]:
                    teacher_paths = train_cfg_dict["policy"]["teacher_model_paths"]
                    for path in teacher_paths:
                        terrain_name = path.split('/')[-2] if '/' in path else "teacher"
                        teacher_tags.append(terrain_name)
                
                wandb_config["tags"] = teacher_tags
                wandb_config["notes"] = f"多教师蒸馏训练 - 学习多地形适应策略"
                
                print(f"🎯 多教师蒸馏模式检测到")
            
            # 初始化wandb
            wandb.init(**wandb_config)
            
            print(f"✅ Wandb initialized successfully!")
            print(f"   Project: {wandb_config['project']}")  
            print(f"   Experiment: {wandb_config['name']}")
            print(f"   URL: {wandb.run.url}")
            print(f"   Log dir: {log_dir}")
            if is_multi_teacher:
                print(f"   Tags: {wandb_config.get('tags', [])}")
            
        except ImportError:
            print("⚠️  Wandb not installed, falling back to tensorboard")
        except Exception as e:
            print(f"⚠️  Wandb initialization failed: {e}")
            print("   Falling back to tensorboard")

# 创建全局任务注册器实例
task_registry = TaskRegistry()
