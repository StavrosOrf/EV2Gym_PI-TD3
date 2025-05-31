import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
import optax
from rl_jax.utils import evaluate_policy


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


def train_ddpg(env, state, key, step_fn, train_steps=1000, buffer_size=1000,  eval_interval=200, batch_size=64, gamma=0.99, tau=0.005):
    obs_size = env.observation_size
    act_size = env.action_size
    actor = DDPGActor(action_dim=act_size)
    critic = DDPGCritic()
    key, subkey = jr.split(key)
    actor_params = actor.init(subkey, jnp.zeros((1, obs_size)))
    key, subkey = jr.split(key)
    critic_params = critic.init(subkey, jnp.zeros(
        (1, obs_size)), jnp.zeros((1, act_size)))
    actor_target = actor_params
    critic_target = critic_params
    actor_opt = optax.adam(1e-4)
    critic_opt = optax.adam(1e-3)
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    from rl_jax.buffer import ReplayBuffer
    buffer = ReplayBuffer(buffer_size)
    for _ in range(buffer_size):
        key, subkey = jr.split(key)
        act = jr.uniform(subkey, shape=(1, act_size), minval=-1.0, maxval=1.0)
        next_state = step_fn(state, act)
        buffer.add((state.obs[0], act[0], next_state.reward[0],
                   next_state.obs[0], next_state.done[0]))
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
        a_t = jax.tree_util.tree_map(
            lambda t, s: t * (1 - tau) + s * tau, a_t, a_p)
        c_t = jax.tree_util.tree_map(
            lambda t, s: t * (1 - tau) + s * tau, c_t, c_p)
        return a_p, c_p, a_t, c_t, a_os, c_os
    for i in range(train_steps):
        batch = [buffer.sample() for _ in range(batch_size)]
        s_b, a_b, r_b, n_b, d_b = map(lambda arr: jnp.array(arr), zip(*batch))
        (actor_params, critic_params, actor_target, critic_target, actor_opt_state, critic_opt_state) = ddpg_update(
            actor_params, critic_params, actor_target, critic_target,
            actor_opt_state, critic_opt_state,
            s_b, a_b, r_b, n_b, d_b
        )
        if i % eval_interval == 0:
            print(f"[DDPG] step={i} — evaluating...")
            evaluate_policy(env, DDPGActor, actor_params)
    return actor_params
