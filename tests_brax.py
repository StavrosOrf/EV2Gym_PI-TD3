import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from brax import envs
from enum import Enum
from flax import linen as nn
import optax
import random
from gym import spaces

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
    """
    Create a Brax environment with batch_size=1, returning:
      - env       : the Brax environment
      - state     : the initial brax.State
      - key       : PRNGKey for additional randomness
      - step_fn   : a JIT-compiled step function
    """
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

class DDPGActor(nn.Module):
    action_dim: int
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return nn.tanh(nn.Dense(self.action_dim)(x))

class DDPGCritic(nn.Module):
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return jnp.squeeze(nn.Dense(1)(x), -1)

class SACActor(nn.Module):
    action_dim: int
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20, 2)
        return mean, log_std

class SACCritic(nn.Module):
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return jnp.squeeze(nn.Dense(1)(x), -1)

# ---------------------
# Training Loop (Supervised-like)
# ---------------------
def train_policy(env_name: BraxEnvName,
                 train_steps: int = 500,
                 buffer_size: int = 100,
                 seed: int = 0):
    env, state, key, step_fn = create_brax_env(env_name, seed)
    obs_size = env.observation_size
    act_size = env.action_size

    policy = Policy(action_dim=act_size)
    key, subkey = jr.split(key)
    params = policy.init(subkey, jnp.zeros((1, obs_size)))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    buffer = ReplayBuffer(buffer_size)
    for _ in range(buffer_size):
        key, subkey = jr.split(key)
        action = jr.uniform(subkey, shape=(1, act_size), minval=-1.0, maxval=1.0)
        next_state = step_fn(state, action)
        buffer.add(state)
        state = next_state

    def loss_fn(p, s):
        obs = s.obs[0]
        action = policy.apply(p, obs[None, ...])[0]
        next_s = step_fn(s, action[None, ...])
        return -next_s.reward[0]

    grad_and_loss = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def update(p, o_s, g):
        updates, o_s = optimizer.update(g, o_s)
        return optax.apply_updates(p, updates), o_s

    for i in range(train_steps):
        s = buffer.sample()
        loss, grads = grad_and_loss(params, s)
        params, opt_state = update(params, opt_state, grads)
        if i % 100 == 0:
            print(f"[Supervised] step={i}, loss={loss:.3f}")

    return params

# ---------------------
# DDPG Training Loop
# ---------------------
def train_ddpg(env_name: BraxEnvName,
               train_steps: int = 1000,
               buffer_size: int = 1000,
               batch_size: int = 64,
               seed: int = 0,
               gamma: float = 0.99,
               tau: float = 0.005):
    env, state, key, step_fn = create_brax_env(env_name, seed)
    obs_size = env.observation_size
    act_size = env.action_size

    actor       = DDPGActor(action_dim=act_size)
    critic      = DDPGCritic()
    key, subkey = jr.split(key)
    actor_params  = actor.init(subkey, jnp.zeros((1, obs_size)))
    key, subkey = jr.split(key)
    critic_params = critic.init(subkey, jnp.zeros((1, obs_size)), jnp.zeros((1, act_size)))
    actor_target  = actor_params
    critic_target = critic_params

    actor_opt       = optax.adam(1e-4)
    critic_opt      = optax.adam(1e-3)
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)

    buffer = ReplayBuffer(buffer_size)
    for _ in range(buffer_size):
        key, subkey = jr.split(key)
        act = jr.uniform(subkey, shape=(1, act_size), minval=-1.0, maxval=1.0)
        next_state = step_fn(state, act)
        buffer.add((state.obs[0], act[0], next_state.reward[0], next_state.obs[0], next_state.done[0]))
        state = next_state

    @jax.jit
    def ddpg_update(a_p, c_p, a_t, c_t, a_os, c_os, s_b, a_b, r_b, n_b, d_b):
        def critic_loss_fn(cp):
            q_val = critic.apply(cp, s_b, a_b)
            next_act = actor.apply(a_t, n_b)
            q_next = critic.apply(c_t, n_b, next_act)
            target = r_b + gamma * (1 - d_b) * q_next
            return jnp.mean((q_val - target) ** 2)

        c_grads = jax.grad(critic_loss_fn)(c_p)
        c_updates, c_os = critic_opt.update(c_grads, c_os, c_p)
        c_p = optax.apply_updates(c_p, c_updates)

        def actor_loss_fn(ap):
            actions = actor.apply(ap, s_b)
            return -jnp.mean(critic.apply(c_p, s_b, actions))

        a_grads = jax.grad(actor_loss_fn)(a_p)
        a_updates, a_os = actor_opt.update(a_grads, a_os, a_p)
        a_p = optax.apply_updates(a_p, a_updates)

        a_t = jax.tree_util.tree_map(lambda t, s: t * (1 - tau) + s * tau, a_t, a_p)
        c_t = jax.tree_util.tree_map(lambda t, s: t * (1 - tau) + s * tau, c_t, c_p)

        return a_p, c_p, a_t, c_t, a_os, c_os

    for i in range(train_steps):
        batch = [buffer.sample() for _ in range(batch_size)]
        s_b, a_b, r_b, n_b, d_b = map(lambda arr: jnp.array(arr), zip(*batch))
        (actor_params, critic_params,
         actor_target, critic_target,
         actor_opt_state, critic_opt_state) = ddpg_update(
            actor_params, critic_params, actor_target, critic_target,
            actor_opt_state, critic_opt_state,
            s_b, a_b, r_b, n_b, d_b
        )
        if i % 100 == 0:
            print(f"[DDPG] step={i}")

    return actor_params

# ---------------------
# SAC Training Loop
# ---------------------
def train_sac(env_name: BraxEnvName,
              train_steps: int = 100000,
              buffer_size: int = 10000,
              batch_size: int = 64,
              seed: int = 0,
              gamma: float = 0.99,
              tau: float = 0.005,
              alpha: float = 0.2,
              eval_interval: int = 200):
    env, state, key, step_fn = create_brax_env(env_name, seed)
    obs_size = env.observation_size
    act_size = env.action_size

    actor   = SACActor(action_dim=act_size)
    critic1 = SACCritic()
    critic2 = SACCritic()
    key, subkey = jr.split(key)
    actor_params   = actor.init(subkey, jnp.zeros((1, obs_size)))
    key, subkey = jr.split(key)
    critic1_params = critic1.init(subkey, jnp.zeros((1, obs_size)), jnp.zeros((1, act_size)))
    key, subkey = jr.split(key)
    critic2_params = critic2.init(subkey, jnp.zeros((1, obs_size)), jnp.zeros((1, act_size)))
    critic1_target = critic1_params
    critic2_target = critic2_params

    actor_opt        = optax.adam(3e-4)
    critic_opt       = optax.adam(3e-4)
    actor_opt_state   = actor_opt.init(actor_params)
    critic1_opt_state = critic_opt.init(critic1_params)
    critic2_opt_state = critic_opt.init(critic2_params)

    buffer = ReplayBuffer(buffer_size)
    for _ in range(buffer_size):
        key, subkey = jr.split(key)
        mean, log_std = actor.apply(actor_params, state.obs[0][None, ...])
        std = jnp.exp(log_std)
        key, subkey = jr.split(key)
        z = jr.normal(subkey, mean.shape)
        action = jnp.tanh(mean + std * z)
        next_state = step_fn(state, action)
        buffer.add((state.obs[0], action[0], next_state.reward[0], next_state.obs[0], next_state.done[0]))
        state = next_state

    @jax.jit
    def sac_update(a_p, c1_p, c2_p, c1_t, c2_t, a_os, c1_os, c2_os, s_b, a_b, r_b, n_b, d_b):
        # Critic loss
        def critic_loss_fn(c1p, c2p):
            q1 = critic1.apply(c1p, s_b, a_b)
            q2 = critic2.apply(c2p, s_b, a_b)
            next_mean, next_log_std = actor.apply(a_p, n_b)
            next_std = jnp.exp(next_log_std)
            key2 = jr.PRNGKey(0)
            z2 = jr.normal(key2, next_mean.shape)
            next_action = jnp.tanh(next_mean + next_std * z2)
            log_prob = jnp.sum(-0.5 * (((z2)**2 + 2*next_log_std + jnp.log(2*jnp.pi))), axis=-1)
            next_q1 = critic1.apply(c1_t, n_b, next_action)
            next_q2 = critic2.apply(c2_t, n_b, next_action)
            next_q = jnp.minimum(next_q1, next_q2) - alpha * log_prob
            target = r_b + gamma * (1 - d_b) * next_q
            loss1 = jnp.mean((q1 - target)**2)
            loss2 = jnp.mean((q2 - target)**2)
            return loss1 + loss2

        (c_loss), (c1_grads, c2_grads) = jax.value_and_grad(critic_loss_fn, argnums=(0,1))(c1_p, c2_p)
        c1_updates, c1_os = critic_opt.update(c1_grads, c1_os)
        c2_updates, c2_os = critic_opt.update(c2_grads, c2_os)
        c1_p = optax.apply_updates(c1_p, c1_updates)
        c2_p = optax.apply_updates(c2_p, c2_updates)

        # Actor loss
        def actor_loss_fn(ap):
            mean, log_std = actor.apply(ap, s_b)
            std = jnp.exp(log_std)
            key3 = jr.PRNGKey(1)
            z3 = jr.normal(key3, mean.shape)
            act3 = jnp.tanh(mean + std * z3)
            logp = jnp.sum(-0.5 * (((z3)**2 + 2*log_std + jnp.log(2*jnp.pi))), axis=-1)
            q1v = critic1.apply(c1_p, s_b, act3)
            q2v = critic2.apply(c2_p, s_b, act3)
            qmin = jnp.minimum(q1v, q2v)
            return jnp.mean(alpha * logp - qmin)

        a_loss, a_grads = jax.value_and_grad(actor_loss_fn)(a_p)
        a_updates, a_os = actor_opt.update(a_grads, a_os)
        a_p = optax.apply_updates(a_p, a_updates)

        # Soft target updates
        c1_t = jax.tree_util.tree_map(lambda t, s: t * (1 - tau) + s * tau, c1_t, c1_p)
        c2_t = jax.tree_util.tree_map(lambda t, s: t * (1 - tau) + s * tau, c2_t, c2_p)

        return a_p, c1_p, c2_p, c1_t, c2_t, a_os, c1_os, c2_os

    for i in range(train_steps):
        batch = [buffer.sample() for _ in range(batch_size)]
        s_b, a_b, r_b, n_b, d_b = map(lambda arr: jnp.array(arr), zip(*batch))
        (actor_params, critic1_params, critic2_params,
         critic1_target, critic2_target,
         actor_opt_state, critic1_opt_state, critic2_opt_state) = sac_update(
            actor_params, critic1_params, critic2_params,
            critic1_target, critic2_target,
            actor_opt_state, critic1_opt_state, critic2_opt_state,
            s_b, a_b, r_b, n_b, d_b
        )
        if i % eval_interval == 0:
            print(f"[SAC] step={i} — evaluating...")
            evaluate_policy(env_name, actor_params, episodes=5)

    return actor_params

# ---------------------
# Evaluation Loop (Metrics Only)
# ---------------------
def evaluate_policy(env_name: BraxEnvName,
                    params,
                    episodes: int = 10,
                    max_steps: int = 200,
                    seed: int = 0):
    """
    Runs:
      • Learned policy (if stochastic: sample from Gaussian + tanh),
      • Random (expert) baseline,
      • Zero-action (simple) baseline.
    Prints mean/std of total episode reward over `episodes`.
    """
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

    # Learned policy sampling (handles both deterministic and stochastic):
    def learned_fn(o):
        # Attempt to unpack (mean, log_std). If no log_std, treat as deterministic.
        out = Policy(action_dim=act_size).apply(params, o[None, ...])
        if isinstance(out, tuple):
            # Stochastic: out = (mean, log_std)
            mean, log_std = out
            std = np.exp(log_std)
            z = np.random.randn(*mean.shape)
            action_batched = np.tanh(mean + std * z)
            return np.array(action_batched)[0]
        else:
            # Deterministic policy
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
    # # 1) Supervised-like training & evaluation
    # print("\n=== Supervised-like Training ===")
    # sup_params = train_policy(BraxEnvName.REACHER)
    # print("\n=== Evaluating Supervised-like Policy ===")
    # evaluate_policy(BraxEnvName.REACHER, sup_params)

    # # 2) DDPG training & evaluation
    # print("\n=== DDPG Training ===")
    # ddpg_params = train_ddpg(BraxEnvName.REACHER)
    # print("\n=== Evaluating DDPG Policy ===")
    # evaluate_policy(BraxEnvName.REACHER, ddpg_params)

    # 3) SAC training & evaluation
    print("\n=== SAC Training ===")
    sac_params = train_sac(BraxEnvName.REACHER)
    print("\n=== Evaluating SAC Policy ===")
    evaluate_policy(BraxEnvName.REACHER, sac_params)
