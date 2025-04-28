import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core_rl
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
from distutils.util import strtobool
class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma = 0.99, lam = 0.95):
        self.obs_buf = np.zeros(core_rl.combined_shape(size, obs_dim), dtype = np.float32)
        self.act_buf = np.zeros(core_rl.combined_shape(size, act_dim), dtype = np.float32)
        self.adv_buf = np.zeros(size, dtype = np.float32)
        self.rew_buf = np.zeros(size, dtype = np.float32)
        self.ret_buf = np.zeros(size, dtype = np.float32)
        self.val_buf = np.zeros(size, dtype = np.float32)
        self.logp_buf = np.zeros(size, dtype = np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        # print("#########ptr, size", self.ptr, self.max_size)
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val = 0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core_rl.discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = core_rl.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = core_rl.statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs = self.obs_buf, act = self.act_buf, adv = self.adv_buf,
                    ret = self.ret_buf, logp = self.logp_buf)
        # print(f"#####act_buf:{self.act_buf}")
        # input()
        return {k : torch.as_tensor(v, dtype = torch.float32) for k, v in data.items()}
    

def ppo(env_fn, actor_critic = core_rl.MLPActorCritic, ac_kwargas = dict(), seed = 0,
        steps_per_epoch = 4000, epochs = 50, gamma = 0.99, clip_ratio = 0.2, pi_lr = 3e-4,
        vf_lr = 1e-3, train_pi_iters = 80, train_v_iters = 80, lam = 0.97, max_ep_len = 1000,
        target_kl = 0.01, save_freq = 10, ent_coef = 0.01, norm_adv = False):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn
    obs_dim = env.single_observation_space.shape
    act_dim = env.single_action_space.shape
    
    ac = actor_critic(obs_dim, env.single_action_space.n, **ac_kwargas)

    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # print(act)
        # input()
        pi, logp, entropy = ac.pi(obs, act)
        if norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() #PyTorch 默认最小化，因此加了负号。
        loss_entropy = entropy.mean()
        loss = loss_pi - ent_coef * loss_entropy
        approx_kl = (logp_old - logp).mean().item()
        return loss, approx_kl
    
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()
    
    pi_optimizer = Adam(ac.pi.parameters(), lr = pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr = vf_lr)

    def update():
        data = buf.get()
        total_loss_pi = 0
        total_loss_v = 0
        total_kl = 0
        train_pi_iters_n = 0
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, kl = compute_loss_pi(data)
            wandb.log({
            "loss_pi": loss_pi,
            "kl": kl,
            })
            total_kl += kl
            total_loss_pi += loss_pi
            train_pi_iters_n += 1
            if kl > 1.5 * target_kl:
                print(f"Early stopping at step {i} due to reaching max kl.")
                break

            loss_pi.backward()
            pi_optimizer.step()

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            total_loss_v += loss_v
            wandb.log({
            "loss_v": loss_v,
            })
            loss_v.backward()
            vf_optimizer.step()
        
        return total_loss_pi / train_pi_iters_n, total_kl / train_pi_iters_n, total_loss_v / train_v_iters

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype = torch.float32))
            next_o, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v, logp)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype = torch.float32))
                buf.finish_path(v)
                if terminal:
                    for item in info:
                        if "episode" in item.keys():
                            print(f"episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")
                            wandb.log({"epoch": epoch, "episodic_return": item['episode']['r'], "episodic_length": item['episode']['l']})
                    # print(f"Episode reward: {ep_ret}, Episode length: {ep_len}")
                o, ep_ret, ep_len = env.reset(), 0, 0
        loss_pi, kl, loss_v = update()
        # wandb.log({
        #     "loss_pi": loss_pi,
        #     "loss_v": loss_v,
        #     "kl": kl,
        # })

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
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type = str, default = 'Overcooked-LLMA-v4')
    parser.add_argument('--hid', type= int, default = 64)
    parser.add_argument('--l', type = int, default = 2)
    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--seed', '-s', type = int, default = 0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp-name', type=str, default='ppo')
    parser.add_argument('--env-reward', type=float, nargs=4,  default=[0.1, 1, 0, 0.001], 
        help='The reward list of the env')
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="twosome_ppo",
        help="the wandb's project name")
    args = parser.parse_args()

    time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    run_name = f"{args.env_id}_task={0}_exp_name={args.exp_name}_seed={args.seed}_{time_str}"
    
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
    
    rewardList = {"subtask finished": args.env_reward[0], "correct delivery": args.env_reward[1], "wrong delivery": -args.env_reward[2], "step penalty": -args.env_reward[3]}
    TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]
    env_params = {'grid_dim': [7,7],
                    'task': TASKLIST[3],
                    'rewardList': rewardList,
                    'map_type': "A",
                    'n_agent': 1,
                    'obs_radius': 2,
                    'mode': "vector",
                    'debug': False
                }

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, False, run_name, env_params) for i in range(1)]
    )
    ppo(envs, actor_critic = core_rl.MLPActorCritic,
        ac_kwargas = dict(hidden_sizes = [args.hid] * args.l), 
        seed = args.seed, steps_per_epoch = args.steps, epochs = args.epochs,
        gamma = args.gamma, max_ep_len = 1000, save_freq = 10, ent_coef = args.ent_coef, norm_adv = args.norm_adv)
