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

from rl_jax.ddpg import DDPGActor, DDPGCritic, train_ddpg
from rl_jax.sac import SACActor, SACCritic, train_sac
from rl_jax.utils import evaluate_policy

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
# Create Brax Environment
# ---------------------
def create_brax_env(env_name: BraxEnvName, seed: int = 0):
    env = envs.create(env_name.value, batch_size=1)
    key = jr.PRNGKey(seed)
    state = env.reset(key)
    step_fn = jax.jit(env.step)
    return env, state, key, step_fn

# ---------------------
# Main Entry
# ---------------------
if __name__ == '__main__':
    # 1) DDPG training & evaluation
    print("\n=== DDPG Training ===")
    env, state, key, step_fn = create_brax_env(BraxEnvName.WALKER2D)
    ddpg_params = train_ddpg(env, state, key, step_fn)

    # 2) SAC training & evaluation
    print("\n=== SAC Training ===")
    env, state, key, step_fn = create_brax_env(BraxEnvName.WALKER2D)
    sac_params = train_sac(env, state, key, step_fn)
