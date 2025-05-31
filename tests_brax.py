import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from brax import envs
from enum import Enum
import optax
import random
from gym import spaces
from flax import linen as nn

from agent.ddpg import DDPGActor, DDPGCritic, train_ddpg
from agent.sac import SACActor, SACCritic, train_sac

# ---------------------
# Environment Names
# ---------------------
class BraxEnvName(Enum):
    ANT = 'ant'
    HALFCHEETAH = 'halfcheetah'
    HUMANOID = 'humanoid'
    WALKER2D = 'walker2d'
    REACHER = 'reacher'

# ---------------------
# Replay Buffer
# ---------------------
class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []

    def add(self, data) -> None:
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(data)

    def sample(self):
        return random.choice(self.buffer)

# ---------------------
# Create Brax Environment
# ---------------------
def create_brax_env(env_name: BraxEnvName, seed: int = 0):
    env = envs.create(env_name.value, batch_size=1)
    key = jr.PRNGKey(seed)
    state = env.reset(key)
    step_fn = jax.jit(env.step)
    return env, state, key, step_fn

# ---------------------
# Policy and Q-Network Definitions
# ---------------------
class Policy(nn.Module):
    action_dim: int
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return nn.tanh(nn.Dense(self.action_dim)(x))

# ---------------------
# Evaluation Loop (Metrics Only)
# ---------------------
def evaluate_policy(env_name: BraxEnvName,
                    params,
                    episodes: int = 10,
                    max_steps: int = 200,
                    seed: int = 0):
    env, _, key, step_fn = create_brax_env(env_name, seed)
    act_size = env.action_size

    def run_policy(policy_fn):
        totals = []
        for _ in range(episodes):
            s = env.reset(key)
            ep_rew = 0.0
            for _ in range(max_steps):
                obs = s.obs[0]
                action = policy_fn(obs)
                action = np.array(action)
                if action.shape != (act_size,):
                    raise ValueError(f"Wrong action shape: {action.shape}")
                s = step_fn(s, action[None, ...])
                ep_rew += float(s.reward[0])
                if bool(s.done[0]):
                    break
            totals.append(ep_rew)
        return np.mean(totals), np.std(totals)

    def learned_fn(o):
        out = Policy(action_dim=act_size).apply(params, o[None, ...])
        if isinstance(out, tuple):
            mean, log_std = out
            std = np.exp(log_std)
            z = np.random.randn(*mean.shape)
            action_batched = np.tanh(mean + std * z)
            return np.array(action_batched)[0]
        else:
            return np.array(out[0])

    def expert_fn(o):
        return np.random.uniform(-1.0, 1.0, size=(act_size,))

    def simple_fn(o):
        return np.zeros((act_size,))

    lm, ls = run_policy(learned_fn)
    em, es = run_policy(expert_fn)
    sm, ss = run_policy(simple_fn)

    print(f"Evaluation ({episodes} episodes):")
    print(f"  Learned: mean = {lm:.2f}, std = {ls:.2f}")
    print(f"  Expert:  mean = {em:.2f}, std = {es:.2f}")
    print(f"  Simple:  mean = {sm:.2f}, std = {ss:.2f}")

# ---------------------
# Main Entry
# ---------------------
if __name__ == '__main__':
    # 1) DDPG training & evaluation
    print("\n=== DDPG Training ===")
    env, state, key, step_fn = create_brax_env(BraxEnvName.REACHER)
    ddpg_params = train_ddpg(env, state, key, step_fn)
    print("\n=== Evaluating DDPG Policy ===")
    evaluate_policy(BraxEnvName.REACHER, ddpg_params)

    # 2) SAC training & evaluation
    print("\n=== SAC Training ===")
    env, state, key, step_fn = create_brax_env(BraxEnvName.REACHER)
    sac_params = train_sac(env, state, key, step_fn)
    print("\n=== Evaluating SAC Policy ===")
    evaluate_policy(BraxEnvName.REACHER, sac_params)