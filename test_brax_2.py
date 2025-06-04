#!/usr/bin/env python3
"""
test_brax_2.py

A minimal, shape‐safe example that uses Brax’s built‐in PPO. 
This version uses the correct import paths for Brax v0.1.x (where
the PPO modules live under brax.training.agents.ppo).
"""

import functools
import jax
import jax.numpy as jnp
from brax import envs

# Import train and networks from brax.training.agents.ppo
from brax.training.agents.ppo.train import train as ppo_train
from brax.training.agents.ppo.networks import make_ppo_networks

def main():
    # ================================
    # Hyperparameters
    # ================================
    ENV_NAME       = "ant"       # Brax environment name
    NUM_ENVS       = 32          # Batch size for training
    NUM_EVAL_ENVS  = 4           # Batch size for evaluation
    SEED           = 0
    NUM_TIMESTEPS  = 200_000     # Total environment steps to collect

    # ================================
    # 1) Factory for a batched training environment
    # ================================
    def make_env(seed):
        return envs.create(env_name=ENV_NAME, batch_size=NUM_ENVS)

    # ================================
    # 2) Factory for a batched evaluation environment
    # ================================
    def make_eval_env(seed):
        return envs.create(env_name=ENV_NAME, batch_size=NUM_EVAL_ENVS)

    # ================================
    # 3) Build our PPO actor–critic network constructor
    # ================================
    # Wrap make_ppo_networks to specify hidden‐layer sizes.
    make_networks = functools.partial(
        make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128),
    )

    # ================================
    # 4) Wrap hyperparameters into ppo_train
    # ================================
    train_fn = functools.partial(
        ppo_train,
        num_timesteps=NUM_TIMESTEPS,
        num_evals=10,
        episode_length=200,
        num_envs=NUM_ENVS,
        num_eval_envs=NUM_EVAL_ENVS,
        batch_size=32,
        num_minibatches=4,
        unroll_length=20,
        num_updates_per_batch=4,
        normalize_observations=True,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        network_factory=make_networks,
        seed=SEED,
    )

    # ================================
    # 5) Call PPO.train with the factories (not raw Env instances)
    # ================================
    make_policy_fn, params, metrics = train_fn(
        environment=make_env,
        eval_env=make_eval_env,
    )

    print("✅ PPO training completed without vmap rank‐0 errors!")

    # ================================
    # 6) (Optional) Run one deterministic evaluation rollout
    # ================================
    key = jax.random.PRNGKey(SEED + 42)
    eval_env = make_eval_env(key)  # Batched eval env
    state = eval_env.reset(key)
    total_return = jnp.zeros((NUM_EVAL_ENVS,))
    done = jnp.zeros((NUM_EVAL_ENVS,), dtype=bool)

    for _ in range(200):
        obs = state.obs                          # (NUM_EVAL_ENVS, obs_dim)
        action, _ = make_policy_fn(params, obs, key)
        # DEBUG: Uncomment to verify shapes:
        # print(">>> DEBUG: eval state.obs.shape =", obs.shape)
        # print(">>> DEBUG: eval action.shape    =", action.shape)
        state = eval_env.step(state, action)
        total_return = total_return + state.reward
        done = jnp.logical_or(done, state.done)
        if jnp.all(done):
            break

    avg_ret = float(jnp.mean(total_return))
    print(f"Deterministic eval average return over {NUM_EVAL_ENVS} envs = {avg_ret:.2f}")

if __name__ == "__main__":
    main()

