import time
import random
import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
from distutils.util import strtobool
import gym
import core

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type = str, default = os.path.basename(__file__).rstrip('.py'),
        help = "the name of the experiment")
    parser.add_argument("--seed", type = int, default = 1,
        help = "random seed")
    parser.add_argument("--cuda", type = lambda x : bool(strtobool(x)), default = True, nargs = "?", const = True,
        help = "whether to use cuda")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="twosome_rl",
        help="the wandb's project name")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    
    #算法参数
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    
    parser.add_argument("--policy-learning-rate", type=float, default=1e-6,
        help="the learning rate of the optimizer")
    parser.add_argument("--value-learning-rate", type=float, default=3e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=16,
        help="the number of mini-batches")
    parser.add_argument("--policy-num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--value-num-minibatches", type=int, default=4,
        help="the number of mini-batches")

    #并行环境 + 多步 rollouts
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=32,
        help="the number of steps to run in each environment per policy rollout")

    #是否开启policy and critic的学习率衰减
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")

    # 用同一批收集到的数据，重复训练策略网络 K 次
    parser.add_argument("--update-epochs", type=int, default=1,
        help="the K epochs to update the policy")
    #--norm-adv 控制要不要对 advantage 做归一化，默认是开启的（True），目的是让训练过程更加稳定、高效。
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    #熵奖励系数
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    #值函数的系数
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    #最大梯度范数
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    #目标 KL 散度
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    #梯度检查点 梯度检查点（Gradient Checkpointing） 是一种用于优化显存使用的技术，特别是在训练深度神经网络时。其核心思想是在训练过程中不保存所有的中间激活值，而是在需要时重新计算某些中间激活，从而节省显存的占用。
    parser.add_argument('--gradient-checkpointing-steps', action='store', type=int, default=8,
        help='The number of steps for gradient checkpointing')
    #ciritic 的 warm up 步数
    parser.add_argument('--critic-warm-up-steps',   action='store', type=int, default=5000, 
        help='The number of time steps to warm up critic')
    
    #环境参数
    parser.add_argument('--env-id', type=str, default='Overcooked-LLMA-v3',
        help='Domain name')
    parser.add_argument('--n-agent', type=int, default=1,
        help='Number of agents')
    parser.add_argument('--grid-dim', type=int, nargs=2, default=[7,7],
        help='Grid world size')
    parser.add_argument('--task', type=int, default=3,
        help='The receipt agent cooks')
    parser.add_argument('--map-type', type=str, default="A",
        help='The type of map')
    parser.add_argument('--obs-radius', type=int, default=2, 
        help='The radius of the agents')
    parser.add_argument('--env-reward', type=float, nargs=4,  default=[0.1, 1, 0, 0.001], 
        help='The reward list of the env')
    parser.add_argument('--mode', type=str, default="vector", 
        help='The type of the observation(vector/image)')    
    parser.add_argument('--debug', type=bool, default=False, 
        help='Whehter print the debug information and render') 
    
     # todo add params
    parser.add_argument('--opt-num-cuda', type=int, default=0,
        help='option a cuda to verify to all tensors to be on the same device')
    parser.add_argument('--llm-base-model', type=str, default="meta-llama/Llama-3.1-8B",
        help='option a cuda to verify to all tensors to be on the same device')
    parser.add_argument('--llm-base-model-path', type=str, default=0,
        help='option a cuda to verify to all tensors to be on the same device')
    parser.add_argument('--record-path', type=str, default="runs",  help='The path to save the tensorboard results')    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.policy_minibatch_size = int(args.batch_size // args.num_minibatches)
    args.value_minibatch_size = int(args.batch_size // args.num_minibatches) 
    return args

def make_env(env_id, seed, idx, capture_video, run_name, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)
        env = MacEnvWrapper(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def ppo(envs, actor_critic = core.MLPActorCritic, ac_kwargs = dict(),
        total_timesteps = 1000000, critic_warm_up_steps = 5000, update_epoch = 1,
        num_envs = 4, num_steps = 32, batch_size = 128, value_minibatch_size = 4, policy_minibatch_size = 32,
        gae_lambda = 0.95, gamma = 0.99, pi_lr = 3e-4,
        vf_lr = 1e-3, args = None, device = "cuda"):
    
    #buffer
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    steps = torch.zeros((num_steps, num_envs)).to(device)

    ##开始训练
    global_step = 0
    pre_global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)

    num_updates = total_timesteps // batch_size
    num_critic_warm_up_updates = critic_warm_up_steps // batch_size

    is_warmup = True

    agent = actor_critic(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    policy_optimizer = Adam(filter(lambda p: p.requires_grad, agent.pi.parameters()), lr = pi_lr)
    value_optimizer = Adam(filter(lambda p: p.requires_grad, agent.v.parameters() ), lr = vf_lr)
    # training
    for update in range(1, 1 + num_updates + num_critic_warm_up_updates):
        if is_warmup and update > num_critic_warm_up_updates:
            is_warmup = False
        
        #学习率退火
        # if args.anneal_lr and not is_warmup:
        #     frac = 1.0 - (update - 1.0 - num_critic_warm_up_updates) / num_updates
        #     policy_optimizer.param_groups[0]["lr"] = frac * pi_lr
        #     value_optimizer.param_groups[0]["lr"] = frac *  vf_lr

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, value, logprob, _ = agent.step(next_obs, is_warmup = is_warmup)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
            
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            steps[step] = torch.Tensor([item['macro_action_steps'] for item in info]).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
            
        #优势函数和returns
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnomterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnomterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                discount = gamma ** steps[t]

                delta = rewards[t] + discount * nextvalues * nextnomterminal - values[t]
                advantages[t] = lastgaelam = delta + discount * gae_lambda * nextnomterminal * lastgaelam
            returns = advantages + values
        
        #flatten the batch
        b_obs = obs.reshape((-1, ) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        kl_explode = False
        policy_update_steps = 0
        pg_loss = torch.tensor(0)
        entropy_loss = torch.tensor(0)
        old_approx_kl = torch.tensor(0)
        approx_kl = torch.tensor(0)
        total_approx_kl = torch.tensor(0)

        for epoch in range(update_epoch):
            if kl_explode:
                break
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, value_minibatch_size):
                end = start + value_minibatch_size
                mb_inds = b_inds[start : end]
                newvalue = agent.get_value(b_obs[mb_inds])

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_clipped, v_loss_unclipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = v_loss * args.vf_coef
                value_optimizer.zero_grad()
                loss.backward()
                value_optimizer.step()
            
            if is_warmup:
                continue

            policy_optimizer.zero_grad()

            for start in range(0, batch_size, policy_minibatch_size):
                if policy_update_steps % args.gradient_checkpointing_steps == 0:
                    total_approx_kl = 0
                policy_update_steps += 1
                end = start + policy_minibatch_size
                mb_inds = b_inds[start:end]

                _, newvalue, newlogprob, entropy = agent.step(b_obs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
              
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    total_approx_kl += approx_kl / args.gradient_checkpointing_steps
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss
                loss /= args.gradient_checkpointing_steps
                
                loss.backward()

                if policy_update_steps % args.gradient_checkpointing_steps == 0:
                    if args.target_kl is not None:
                        if total_approx_kl > args.target_kl:
                            policy_optimizer.zero_grad()
                            kl_explode = True
                            policy_update_steps -= args.gradient_checkpointing_steps
                            break
                    
                    policy_optimizer.step()
                    policy_optimizer.zero_grad()
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/policy_learning_rate", policy_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/value_learning_rate", value_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/total_approx_kl", total_approx_kl.item(), global_step)
        writer.add_scalar("losses/policy_update_times", policy_update_steps // args.gradient_checkpointing_steps, global_step)
        print("SPS:", global_step, (time.time() - start_time))
        writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)

    #     if global_step // 10000 != pre_global_step // 10000: 
    #         agent.save(global_step // 10000, f"{args.record_path}/{args.exp_name}/{run_name}/{args.save_path}")
    #     pre_global_step = global_step

    # agent.save(global_step // 10000 + 1, f"{args.record_path}/{args.exp_name}/{run_name}/{args.save_path}")
    envs.close()
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    run_name = f"{args.env_id}_task={args.task}_exp_name={args.exp_name}_seed={args.seed}_{time_str}"
    if args.track:
        import wandb
        wandb.init(
            project = args.wandb_project_name,
            sync_tensorboard = True,
            config = vars(args),
            name = run_name,
            monitor_gym = True,
            save_code = True,
        )

    #记录实验的超参数
    writer = SummaryWriter(f"{args.record_path}/{args.exp_name}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cpu"
    if args.cuda == True:
        device = torch.device(f"cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #0.1, 1, 0, 0.001
    rewardList = {"subtask finished": args.env_reward[0], "correct delivery": args.env_reward[1], "wrong delivery": -args.env_reward[2], "step penalty": -args.env_reward[3]}
    TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]
    env_params = {'grid_dim': args.grid_dim,
                    'task': TASKLIST[args.task],
                    'rewardList': rewardList,
                    'map_type': args.map_type,
                    'n_agent': args.n_agent,
                    'obs_radius': args.obs_radius,
                    'mode': args.mode,
                    'debug': args.debug
                }

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, env_params) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    ppo(envs, actor_critic = core.MLPActorCritic,
        gae_lambda = args.gae_lambda, gamma = args.gamma,
        pi_lr = args.policy_learning_rate, vf_lr = args.value_learning_rate,
        total_timesteps = args.total_timesteps, critic_warm_up_steps = args.critic_warm_up_steps, update_epoch = args.update_epochs,
        num_envs = args.num_envs, num_steps = args.num_steps, batch_size = args.batch_size, 
        value_minibatch_size = args.value_minibatch_size, policy_minibatch_size = args.policy_minibatch_size,
        args = args, device = device)
