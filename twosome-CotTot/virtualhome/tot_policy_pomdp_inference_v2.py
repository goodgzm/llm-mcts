# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import copy
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import virtual_home 
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
from tot_policy_pomdp_v2 import LLMAgent, LanguageNode


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
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
    parser.add_argument("--depth", type=int, default=20, help="the depth of the MCTS")

    args = parser.parse_args()
    return args


def make_env(env_id, seed, idx, capture_video, run_name, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)

        # env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
    time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(time.time()))
    run_name = f"{args.env_id}__task={args.exp_name}__seed={args.seed}__{time_str}__llm={args.llm_base_model.split('/')[-1]}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"{args.record_path}/{args.exp_name}/{run_name}")
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

    env_params = {
        'seed': args.seed,
        'debug': args.debug,
    }

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, env_params) for i in
         range(args.num_envs)]
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
    # set agent and LLM model
    agent = LLMAgent(task=args.task, normalization_mode=args.normalization_mode,device=device, llm_base_model=args.llm_base_model, llm_base_model_path=args.llm_base_model_path)
    # agent = LLMAgent(task=args.task)

    for path_num in range(0, args.path_num):
        next_obs = torch.Tensor(envs.reset()).to(device)
        root = LanguageNode(state=next_obs, task=args.env_id, env=envs)
        agent.expand(next_obs, root)
        nodes = [child for child in root.children]
        done = False
        step = 0

        # BFS
        while not done and step < args.depth:
            expanded_nodes = []
            for node in nodes:
                expanded_nodes.append(node)
                done = done or node.done
            next_nodes = agent.select(expanded_nodes, args.expanded_num)
            new_nodes = []
            for next_node in next_nodes:
                
                agent.expand(next_node.state, next_node, is_stochastic = bool(random.random() < args.stochastic))
                for child in next_node.children:
                    new_nodes.append(child)
            nodes = new_nodes
            step += 1

        # MC update
        print(f"global_step={global_step}, num_path = {path_num}, episodic_return={rewards.sum()}, episodic_length={step}")
        writer.add_scalar("charts/episodic_return", rewards.sum(), global_step)
        writer.add_scalar("charts/episodic_length", step, global_step)

    envs.close()
    writer.close()