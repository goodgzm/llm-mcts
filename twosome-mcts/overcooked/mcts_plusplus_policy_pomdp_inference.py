# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import copy
import os
import random
import time
from collections import defaultdict
from distutils.util import strtobool

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
from mcts_plusplus_policy_pomdp import LLMAgent, LanguageNode, ActionNode


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=32,
        help="the number of steps to run in each environment per policy rollout")

    #env_parameter
    parser.add_argument('--env-id',                 action='store',        type=str,             default='Overcooked-LLMA-v3',  help='Domain name')
    parser.add_argument('--n-agent',                action='store',        type=int,             default=1,                     help='Number of agents')
    parser.add_argument('--grid-dim',               action='store',        type=int,   nargs=2,  default=[7,7],                 help='Grid world size')
    parser.add_argument('--task',                   action='store',        type=int,             default=3,                     help='The receipt agent cooks')
    parser.add_argument('--map-type',               action='store',        type=str,             default="A",                   help='The type of map')
    parser.add_argument('--obs-radius',             action='store',        type=int,             default=2,                     help='The radius of the agents')
    parser.add_argument('--env-reward',             action='store',        type=float, nargs=4,  default=[0.1, 1, 0, 0.001],    help='The reward list of the env')
    parser.add_argument('--mode',                   action='store',        type=str,             default="vector",              help='The type of the observation(vector/image)')    
    parser.add_argument('--debug',                  action='store',        type=bool,            default=False,                 help='Whehter print the debug information and render') 
    
    

    parser.add_argument('--save-path',              action='store',        type=str,             default="saved_models",        help='The path to save the checkpoint')
    parser.add_argument('--save-interval',          action='store',        type=int,             default=10,                    help='The interval for saving model for certain num_updates')
    parser.add_argument('--record-path',            action='store',        type=str,             default="llm5_runs",           help='The path to save the tensorboard results')

    parser.add_argument('--normalization-mode',     action='store',        type=str,             default="token",               help='The normalization mode of how to deal with the logits of each token')    

    # todo add params
    parser.add_argument('--opt-num-cuda',     action='store',        type=int,    default=0,               help='option a cuda to verify to all tensors to be on the same device')
    parser.add_argument('--llm-base-model',     action='store',        type=str,    default="meta-llama/Llama-3.1-8B",               help='option a cuda to verify to all tensors to be on the same device')
    parser.add_argument('--llm-base-model-path',     action='store',        type=str,    default=0,               help='option a cuda to verify to all tensors to be on the same device')
    parser.add_argument('--stochastic',     action='store',        type=float,    default=0.2,               help='option a cuda to verify to all tensors to be on the same device')
    parser.add_argument('--value-weight',     action='store',        type=float,    default=0.5,               help='option a cuda to verify to all tensors to be on the same device')
    parser.add_argument('--path-num',     action='store',        type=int,    default=500,               help='option a cuda to verify to all tensors to be on the same device')

    parser.add_argument('--rnd', action='store',type = bool,default = False, help = 'if use rnd')
    parser.add_argument('--rnd-weight',     action='store',        type=float,    default=0.5,               help='option a cuda to verify to all tensors to be on the same device')
    parser.add_argument('--transpositions', action='store',type = bool, default = False, help = 'if use transopositions')
    parser.add_argument("--depth", type=int, default=20, help="the depth of the MCTS")
    args = parser.parse_args()
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


if __name__ == "__main__":
    """ init params """
    args = parse_args()
    time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(time.time()))
    run_name = f"{args.env_id}__task={args.exp_name}__stochastic={args.stochastic}__seed={args.seed}__{time_str}__llm={args.llm_base_model.split('/')[-1]}__normalization_mode={args.normalization_mode}__value_weight={args.value_weight}__path_num={args.path_num}__is_rnd={args.rnd}__transpositions={args.transpositions}__depth={args.depth}__MCTS++"
    # if args.track:
    #     import wandb
    #
    #     wandb.init(
    #         project=args.wandb_project_name,
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         monitor_gym=True,
    #         save_code=True,
    #     )
    writer = SummaryWriter(f"{args.record_path}/{args.exp_name}/{run_name}")
    rnd_writer = SummaryWriter(f"{args.record_path}/{args.exp_name}/{run_name}/rnd_log")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # If you're using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    """ set task information and create gym environment """
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


    """ start training """
    # ALGO Logic: Storage setup
    # obs: 存储每个环境每个步骤的 观察，single_observation_space 是来自环境中观察数据
    # actions：存储每个环境每个步骤的 动作，single_action_space 是动作空间
    # logprobs：存储每个环境每个步骤的 动作对数概率
    # rewards：存储每个环境每个步骤的 执行动作后获得的奖励
    # dones：存储每个环境每个步骤的 终止标志，为 1 时就是达到终止条件
    # values：存储每个环境每个步骤的 价值估计；critic 用于估计当前价值，即从该状态开始智能体在当前策略下可以预期获得的累计奖励
    # steps：存储每个环境每个步骤的 已经进行的步数
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    steps = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    pre_global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # traj: MCTS 中每条路径的节点
    # traj_reward: MCTS 中每条路径的奖励
    traj = []
    traj_rewards = []
    # set agent and LLM model
    agent = LLMAgent(task=args.task, normalization_mode=args.normalization_mode, device=device,
                     rnd=args.rnd, tb_logger=rnd_writer, value_weight=args.value_weight, rnd_weight=args.rnd_weight,
                     llm_base_model=args.llm_base_model, llm_base_model_path=args.llm_base_model_path)

    done = next_done
    # 建立 MCTS 树根节点
    root = LanguageNode(state=next_obs, initial_value=1, task=args.env_id)
    agent.expand(next_obs, root, envs)
    # modify
    tree_node = defaultdict(list)
    rnd_train_data_cnt = 0
    for path_num in range(0, args.path_num):
        node_path = []
        # root._visit_count += 1
        node = root
        done = torch.zeros(args.num_envs).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        step = 0
        next_obs = torch.Tensor(envs.reset()).to(device)
        envs.reset()

        # 探索路径 探索到 args.depth 步或探索到任务完成条件终止
        while not done and step < args.depth:
            #  当前节点没有结束任务，且其子节点是叶子节点（还未被扩展），扩展子节点
            if node.is_leaf():
                _ = agent.expand(next_obs, node, envs,False)
            assert(type(node) == LanguageNode)
            node._visit_count += 1

            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            next_obs_temp = next_obs
            envs_temp = copy.deepcopy(envs)

            action, value, next_node, action_name = agent.select(node)
            next_action_node = next_node
            # 更新路径和当前所在节点
            node_path.append(next_node)

            values[step] = value.flatten()
            actions[step] = action

            next_obs, reward, done, info = envs.step(action.cpu().numpy())

            # stochastic situation
            if 'chop' in action_name and random.random() < args.stochastic:
                next_obs = next_obs_temp
                envs = envs_temp
                reward = -0.001
                done = next_done

            # modify:
            next_node = agent.expand(next_obs, next_node, envs, type(next_node) == ActionNode) # expand and select a state node
            if args.rnd:
                agent.collect_data(next_obs)
                rnd_train_data_cnt += 1

            # 奖励与状态更新
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            steps[step] = torch.Tensor([item['macro_action_steps'] for item in info]).to(device)

            # modify
            if args.transpositions:
                if next_action_node not in tree_node[action_name]:
                    tree_node[action_name].append(next_action_node)
            # 更新路径和当前所在节点
            node = next_node
            step += 1

        # modify
        if args.transpositions:
            agent.transpositions_update(node_path, rewards[:step], tree_node)
        # MC update
        agent.mcts_update(node_path, rewards[:step])

        traj.append(node_path)
        traj_rewards.append(rewards[:step])
        print(f"global_step={global_step}, num_path = {path_num}, episodic_return={rewards.sum()}, episodic_length={step}")
        writer.add_scalar("charts/episodic_return", rewards.sum(), global_step)
        writer.add_scalar("charts/episodic_length", step, global_step)
        if args.rnd and rnd_train_data_cnt > 15:
            agent.train()

    rewards_sum = torch.tensor([i.sum() for i in traj_rewards]).view(-1)
    num_success = (rewards_sum >= 1).sum().item()
    writer.add_text("scalrs/num_success", str(num_success), global_step=0)

    envs.close()
    writer.close()