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