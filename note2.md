# play记录
```python
ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root = log_pth, 
        env=env, 
        name=args.task, 
        args=args
    )

In task_registry.make_alg_runner: ppo_runner实际为OnPolicyRunner的一个实例
runner = OnPolicyRunner(
                env=env,
                train_cfg=train_cfg_dict,
                log_dir=log_dir,
                device=args.rl_device if args is not None else "cuda:0",
                **kwargs
            )

In play:

#获得推理策略，policy是ActorCriticRMA的一个实例
policy = ppo_runner.get_inference_policy(device=env.device)

obs的处理应该和actor_critic等文件没关系，主要在play里的关于obs部分查就行
actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

obs = env.get_observations()  # 获取初始观测，观测应该就是直接从get_observations里得到的



student_actions = self.actor_critic.action_mean
teacher_actions = self.actor_critic.get_teacher_actions(
                obs_batch, 
                terrain_ids=terrain_ids,  # 使用地形ID选择对应教师
                hist_encoding=True,  # 与play.py保持一致
                env=self.env  # 传入环境实例进行真实处理
            )
student_processed = self.process_actions(student_actions)  # 学生动作通过process_actions处理
teacher_processed = self.process_actions(teacher_actions)  # 教师动作通过process_actions处理
behavior_cloning_loss = F.mse_loss(student_processed, teacher_processed)

def process_actions(self, raw_actions):
        """处理动作，与humanoid_robot中的动作处理逻辑相同
        
        Args:
            raw_actions (torch.Tensor): 原始动作
            
        Returns:
            torch.Tensor: 处理后的动作（与humanoid_robot.step()方法中self.actions相同）
        """
        clip_actions = self.clip_actions / self.action_scale
        processed_actions = torch.clip(raw_actions, -clip_actions, clip_actions).to(self.device)
        return processed_actions


# 地形：
for item in combine_config.proportions:
            terrain_type, index, weight = item
            id = index
   terrain.idx = id 
   self.terrain_type[i, j] = terrain.idx: self = Terrain

   obs_buf.shape() = [num_envs，D]

   已知：每一格子（行，列）对应的地形id，对应于proportion中的id
   obs和地形id可在rollout过程中通过step()被采样
   问题：怎么先把obs和地形id对应起来，obs_buf.shape() = [num_envs，D]，需要知道行和列是怎么并为num_envs的，需要将Terrain.terrain_type[i, j]这个两维张量reshape成num_envs的一个一维张量，把这个信息添加到extras中
   Terrain.terrain_type[i, j]是固定的，但每个机器人的位置是不完全确定的
   

——>
    把extras中的地形id信息存到buffer中
——>
   从buffer里取出一个batch的obs和对应的地形id

   修改storage和transition类，加入地形信息，minibatch返回带地形信息
——>
    返回的地形信息为一个[num_env]的一维向量，不需要_extract_terrain_ids

   _extract_terrain_ids()填充最终蒸馏所需的terrain_ids
   最终希望：对一个batch中的每一个obs，找到相应的地形id