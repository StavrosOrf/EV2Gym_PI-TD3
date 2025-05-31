import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
import optax

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

def train_sac(env, state, key, step_fn, train_steps=100000, buffer_size=10000, batch_size=64, gamma=0.99, tau=0.005, alpha=0.2, eval_interval=200):
    obs_size = env.observation_size
    act_size = env.action_size
    actor = SACActor(action_dim=act_size)
    critic1 = SACCritic()
    critic2 = SACCritic()
    key, subkey = jr.split(key)
    actor_params = actor.init(subkey, jnp.zeros((1, obs_size)))
    key, subkey = jr.split(key)
    critic1_params = critic1.init(subkey, jnp.zeros((1, obs_size)), jnp.zeros((1, act_size)))
    key, subkey = jr.split(key)
    critic2_params = critic2.init(subkey, jnp.zeros((1, obs_size)), jnp.zeros((1, act_size)))
    critic1_target = critic1_params
    critic2_target = critic2_params
    actor_opt = optax.adam(3e-4)
    critic_opt = optax.adam(3e-4)
    actor_opt_state = actor_opt.init(actor_params)
    critic1_opt_state = critic_opt.init(critic1_params)
    critic2_opt_state = critic_opt.init(critic2_params)
    from tests_brax import ReplayBuffer
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
        c1_t = jax.tree_util.tree_map(lambda t, s: t * (1 - tau) + s * tau, c1_t, c1_p)
        c2_t = jax.tree_util.tree_map(lambda t, s: t * (1 - tau) + s * tau, c2_t, c2_p)
        return a_p, c1_p, c2_p, c1_t, c2_t, a_os, c1_os, c2_os
    for i in range(train_steps):
        batch = [buffer.sample() for _ in range(batch_size)]
        s_b, a_b, r_b, n_b, d_b = map(lambda arr: jnp.array(arr), zip(*batch))
        (actor_params, critic1_params, critic2_params, critic1_target, critic2_target, actor_opt_state, critic1_opt_state, critic2_opt_state) = sac_update(
            actor_params, critic1_params, critic2_params,
            critic1_target, critic2_target,
            actor_opt_state, critic1_opt_state, critic2_opt_state,
            s_b, a_b, r_b, n_b, d_b
        )
        if i % eval_interval == 0:
            print(f"[SAC] step={i} — evaluating...")
    return actor_params