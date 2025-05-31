import numpy as np

def evaluate_policy(env, Policy, params, episodes=10, max_steps=20000, seed=0):
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    key = jr.PRNGKey(seed)
    act_size = env.action_size
    step_fn = jax.jit(env.step)

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