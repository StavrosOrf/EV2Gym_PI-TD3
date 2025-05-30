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
    def add(self, state) -> None:
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(state)
    def sample(self):
        return random.choice(self.buffer)

# ---------------------
# Create Brax Environment
# ---------------------
def create_brax_env(env_name: BraxEnvName, seed: int = 0):
    """
    Create a Brax environment with batch_size=1, return env, initial state, RNG key, and JIT-compiled step.
    """
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
# Training Loop
# ---------------------
def train_policy(env_name: BraxEnvName,
                 train_steps: int = 50000,
                 buffer_size: int = 10000,
                 seed: int = 0):
    # Initialize environment
    env, state, key, step_fn = create_brax_env(env_name, seed)
    obs_size = env.observation_size
    act_size = env.action_size

    # Initialize policy network and optimizer
    policy = Policy(action_dim=act_size)
    key, subkey = jr.split(key)
    params = policy.init(subkey, jnp.zeros((1, obs_size)))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    # Populate replay buffer with random transitions
    buffer = ReplayBuffer(buffer_size)
    for _ in range(buffer_size):
        key, subkey = jr.split(key)
        action = jr.uniform(subkey, shape=(1, act_size), minval=-1.0, maxval=1.0)
        state = step_fn(state, action)
        buffer.add(state)

    # Define loss function
    def loss_fn(p, s):
        obs = s.obs[0]
        act = policy.apply(p, obs[None, ...])[0]
        next_s = step_fn(s, act[None, ...])
        return -next_s.reward[0]

    # JIT-compiled gradient
    grad_and_loss = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def update(p, o_s, g):
        updates, o_s = optimizer.update(g, o_s)
        p = optax.apply_updates(p, updates)
        return p, o_s

    # Training iterations
    for i in range(train_steps):
        s = buffer.sample()
        loss, grads = grad_and_loss(params, s)
        params, opt_state = update(params, opt_state, grads)
        if i % 100 == 0:
            print(f"[Train] step={i}, loss={loss:.3f}")

    print(f"Training complete for {env_name.value}.")
    return params

# ---------------------
# Evaluation Loop (Metrics Only)
# ---------------------
def evaluate_policy(env_name: BraxEnvName,
                    params,
                    episodes: int = 30,
                    max_steps: int = 200,
                    seed: int = 0):
    """
    Evaluate learned policy against expert (random) and simple (zero) baselines.
    Prints average total rewards for each over multiple episodes.
    """
    env, _, key, step_fn = create_brax_env(env_name, seed)
    act_size = env.action_size

    def run_policy(policy_fn):
        totals = []
        for _ in range(episodes):
            s = env.reset(key)
            ep_reward = 0.0
            for _ in range(max_steps):
                obs = s.obs[0]
                action = policy_fn(obs)
                # ensure correct shape
                action = np.array(action)
                if action.shape != (act_size,):
                    raise ValueError(f"Action has wrong shape {action.shape}, expected {(act_size,)}")
                s = step_fn(s, action[None,...])
                ep_reward += float(s.reward[0])
                if bool(s.done[0]):
                    break
            totals.append(ep_reward)
        return np.mean(totals), np.std(totals)

    # Learned policy function
    def learned_fn(o):
        return np.array(Policy(action_dim=act_size).apply(params, o[None,...])[0])
    # Expert baseline: random actions
    def expert_fn(o):
        return np.random.uniform(-1.0, 1.0, size=(act_size,))
    # Simple baseline: zero actions
    def simple_fn(o):
        return np.zeros((act_size,))

    # Evaluate
    lm, ls = run_policy(learned_fn)
    em, es = run_policy(expert_fn)
    sm, ss = run_policy(simple_fn)

    print(f"Evaluation over {episodes} episodes:")
    print(f"Learned: mean={lm:.2f}, std={ls:.2f}")
    print(f"Expert:  mean={em:.2f}, std={es:.2f}")
    print(f"Simple:  mean={sm:.2f}, std={ss:.2f}")

# ---------------------
# Main Entry
# ---------------------
if __name__ == '__main__':
    # Train on Reacher environment
    trained_params = train_policy(BraxEnvName.REACHER)
    # Evaluate and compare
    evaluate_policy(
        BraxEnvName.REACHER,
        trained_params,
        episodes=10
    )
    
    # Train on Walker environment
    trained_params = train_policy(BraxEnvName.WALKER)
    # Evaluate and compare
    evaluate_policy(
        BraxEnvName.WALKER,
        trained_params,
        episodes=10
    )

