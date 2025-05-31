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
    env = envs.create(env_name.value, batch_size=1)
    key = jr.PRNGKey(seed)
    state = env.reset(key)
    step_fn = jax.jit(env.step)
    return env, state, key, step_fn

# ---------------------
# Policy Network Definition
# ---------------------
class Policy(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return nn.tanh(x)

# ---------------------
# Q-network for DDPG
# ---------------------
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x)

# ---------------------
# Supervised-like Training Loop
# ---------------------
def train_policy(env_name: BraxEnvName,
                 train_steps: int = 50000,
                 buffer_size: int = 1000,
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
        state = step_fn(state, action)
        buffer.add(state)

    def loss_fn(p, s):
        obs = s.obs[0]
        act = policy.apply(p, obs[None, ...])[0]
        next_s = step_fn(s, act[None, ...])
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
               train_steps: int = 10000,
               buffer_size: int = 10000,
               batch_size: int = 64,
               seed: int = 0,
               gamma: float = 0.99,
               tau: float = 0.005):
    env, state, key, step_fn = create_brax_env(env_name, seed)
    obs_size = env.observation_size
    act_size = env.action_size

    # Initialize actor and critic networks
    actor = Policy(action_dim=act_size)
    critic = QNetwork()
    key, subkey = jr.split(key)
    actor_params = actor.init(subkey, jnp.zeros((1, obs_size)))
    key, subkey = jr.split(key)
    critic_params = critic.init(subkey, jnp.zeros((1, obs_size)), jnp.zeros((1, act_size)))
    # Target networks
    actor_target = actor_params
    critic_target = critic_params

    # Optimizers and states
    actor_opt = optax.adam(1e-4)
    critic_opt = optax.adam(1e-3)
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)

    buffer = ReplayBuffer(buffer_size)
    # Fill buffer
    for _ in range(buffer_size):
        key, subkey = jr.split(key)
        act = jr.uniform(subkey, shape=(1, act_size), minval=-1.0, maxval=1.0)
        state = step_fn(state, act)
        buffer.add((state.obs[0], act[0], state.reward[0], state.obs[0], state.done[0]))

    # DDPG update with opt_states
    @jax.jit
    def ddpg_update(actor_p, critic_p, actor_t, critic_t, a_os, c_os, s_b, a_b, r_b, n_b, d_b):
        # Critic loss and update
        def critic_loss_fn(cp):
            q_val = critic.apply(cp, s_b, a_b)
            next_act = actor.apply(actor_t, n_b)
            q_next = critic.apply(critic_t, n_b, next_act)
            target = r_b + gamma * (1 - d_b) * q_next
            return jnp.mean((q_val - target)**2)
        critic_grads = jax.grad(critic_loss_fn)(critic_p)
        critic_updates, c_os = critic_opt.update(critic_grads, c_os, critic_p)
        critic_p = optax.apply_updates(critic_p, critic_updates)

        # Actor loss and update
        def actor_loss_fn(ap):
            actions = actor.apply(ap, s_b)
            return -jnp.mean(critic.apply(critic_p, s_b, actions))
        actor_grads = jax.grad(actor_loss_fn)(actor_p)
        actor_updates, a_os = actor_opt.update(actor_grads, a_os, actor_p)
        actor_p = optax.apply_updates(actor_p, actor_updates)

        # Soft update targets
        actor_t = jax.tree_util.tree_map(lambda tp, p: tp * (1 - tau) + p * tau, actor_t, actor_p)
        critic_t = jax.tree_util.tree_map(lambda tp, p: tp * (1 - tau) + p * tau, critic_t, critic_p)

        return actor_p, critic_p, actor_t, critic_t, a_os, c_os

    # Training
    for i in range(train_steps):
        batch = [buffer.sample() for _ in range(batch_size)]
        s_b, a_b, r_b, n_b, d_b = map(lambda arr: jnp.array(arr), zip(*batch))
        actor_params, critic_params, actor_target, critic_target, actor_opt_state, critic_opt_state = \
            ddpg_update(actor_params, critic_params, actor_target, critic_target,
                        actor_opt_state, critic_opt_state,
                        s_b, a_b, r_b, n_b, d_b)
        if i % 100 == 0:
            print(f"[DDPG] step={i}")

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
    Evaluate learned, expert (random), and simple (zero) policies.
    Prints mean and std of total rewards.
    """
    env, _, key, step_fn = create_brax_env(env_name, seed)
    act_size = env.action_size

    def run_policy(policy_fn):
        rewards = []
        for _ in range(episodes):
            s = env.reset(key)
            ep_rew = 0.0
            for _ in range(max_steps):
                obs = s.obs[0]
                action = policy_fn(obs)
                action = np.array(action)
                if action.shape != (act_size,):
                    raise ValueError(f"Wrong action shape: {action.shape}")
                s = step_fn(s, action[None,...])
                ep_rew += float(s.reward[0])
                if bool(s.done[0]):
                    break
            rewards.append(ep_rew)
        return np.mean(rewards), np.std(rewards)

    # Learned policy
    def learned_fn(o):
        return np.array(Policy(action_dim=act_size).apply(params, o[None,...])[0])
    # Expert baseline (random)
    def expert_fn(o):
        return np.random.uniform(-1.0, 1.0, size=(act_size,))
    # Simple baseline (zero)
    def simple_fn(o):
        return np.zeros((act_size,))

    lm, ls = run_policy(learned_fn)
    em, es = run_policy(expert_fn)
    sm, ss = run_policy(simple_fn)

    print(f"Evaluation ({episodes} eps):")
    print(f"Learned: mean={lm:.2f}, std={ls:.2f}")
    print(f"Expert:  mean={em:.2f}, std={es:.2f}")
    print(f"Simple:  mean={sm:.2f}, std={ss:.2f}")

# ---------------------
# Main Entry
# ---------------------
if __name__ == '__main__':
    # Supervised-like training
    sup_params = train_policy(BraxEnvName.REACHER)
    # DDPG training
    ddpg_params = train_ddpg(BraxEnvName.REACHER)

    print("=== Evaluating Supervised-like Policy ===")
    evaluate_policy(
        BraxEnvName.REACHER,
        sup_params,
        episodes=10
    )

    print("=== Evaluating DDPG Policy ===")
    evaluate_policy(
        BraxEnvName.REACHER,
        ddpg_params,
        episodes=10
    )
