import torch

def td_lambda_forward_view(
    rewards, dones, states, actions, critic, gamma=0.99, lambda_=0.95, horizon = -1
):
    """
    Implements TD(lambda) as in the formula:
        G_t^λ = (1-λ) * sum_{n=1}^{T-t-1} λ^{n-1} G_t^{(n)} + λ^{T-t-1} G_t^{(T-t)}
    for all t in [0, horizon-1], batched.
    - rewards: [B, H]
    - dones:   [B, H] (1 if done, 0 otherwise)
    - states:  [B, H+1, state_dim]
    - actions: [B, H+1, action_dim]
    - critic: function(states, actions) -> Q-values [B, H+1]
    Returns:
    - td_lambda: [B, H]
    """
    B, H = rewards.shape
    device = rewards.device
    

    assert horizon < H and horizon > 1, "Horizon must be less than or equal to H and greater than 0."
    rewards = rewards[:, :horizon]
    dones = dones[:, :horizon]
    states = states[:, :horizon + 1, :]
    actions = actions[:, :horizon + 1, :]
    H = horizon
    
    # print(f"\n\nTD(lambda) forward view: B={B}, H={H}, device={device}")
    # print(f"Rewards shape: {rewards.shape}\nDones shape: {dones.shape}"
    #       f"\nStates shape: {states.shape}\nActions shape: {actions.shape}")

    # Compute all Q(s_{t+n}, a_{t+n}) up to H (for bootstrapping)
    with torch.no_grad():
        
        q_bootstrap = critic.Q1(states.reshape(-1, states.shape[-1]),
                             actions.reshape(-1, actions.shape[-1])
                             )
        
        # ifwe want to average Q-values from both critics
        # q1, q2 = critic(states.reshape(-1, states.shape[-1]),
        #                      actions.reshape(-1, actions.shape[-1])
        #                      )
        # q_bootstrap = (q1 + q2) / 2.0  # Average Q-values from both critics
        
        q_bootstrap = q_bootstrap.view(B, H+1)  # Reshape to [B, H+1]    
        
        # print(f"Q-values computed: {q_bootstrap.shape}, first 5 values: {q_bootstrap[:, :5]}")
        # print(f"Q-values shape: {q_bootstrap.shape}")

    td_lambda = torch.zeros(B, H, device=device)

    # For each t, compute all G_t^{(n)}
    for t in range(H):
        # Maximum possible n: until end or done
        G_lambda = torch.zeros(B, device=device)
        # print(f"Processing t={t}: G_lambda shape: {G_lambda.shape}")
        # For each possible n-step return
        for n in range(1, H-t+1):
            # Rewards from t to t+n-1
            idxs = torch.arange(t, t+n, device=device)
            gammas = gamma ** torch.arange(n, device=device)
            reward_slice = rewards[:, idxs]    # [B, n]
            done_slice = dones[:, idxs]        # [B, n]
            # print(f"Processing n={n}: reward_slice shape: {reward_slice.shape}, done_slice shape: {done_slice.shape}")
            # Cumulative done mask: 1 until first done (0 after)
            mask = torch.cumprod(1 - done_slice, dim=1)
            # For correct bootstrapping, only up to first done
            mask = torch.cat([torch.ones(B,1,device=device), mask[:,:-1]], dim=1)
            reward_sum = torch.sum(reward_slice * gammas * mask, dim=1)   # [B]

            # Bootstrap Q for G_t^{(n)}
            q_val = torch.zeros(B, device=device)
            # If not done before t+n, add Q
            not_done_n = torch.prod(1 - done_slice, dim=1)
            if t + n < H + 1:            
                q_val = not_done_n * (gamma ** n) * q_bootstrap[:, t+n]
            G_n = reward_sum + q_val

            # λ weights
            if n < H-t:
                weight = (1 - lambda_) * (lambda_ ** (n-1))
            else:
                weight = lambda_ ** (n-1)
            G_lambda = G_lambda + weight * G_n

        td_lambda[:, t] = G_lambda

    # print(f"Final TD(lambda) shape: {td_lambda.shape}, first 5 values: {td_lambda[:, :5]}")
    return td_lambda[:,0]
