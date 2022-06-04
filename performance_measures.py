from tic_env import TictactoeEnv, OptimalPlayer

"""
M_opt measures the performance of pi against the optimal policy. 
To compute M_opt, we run pi against Opt(0) for N = 500 games for different random seeds. 
pi makes the 1st move in 250 games, and Opt(0) makes the 1st move in the rest. 
We count how many games pi wins (N_win) and loses (N_loss) and define M_opt = N_win - N_loss / N.

M_rand measures the performance of against the random policy. 
To compute M_rand, we repeat what we did for computing M_opt but by using Opt(1) instead of Opt(0).
"""

def M(policy, epsilon, N):
    """Compute the performance of the policy against the an optimal player with a given epsilon
    
    Parameters
    ----------
    policy :
        The policy to compute the performance of, should have an act method
    epsilon : float
        The epsilon value to use for the optimal player
    N : int
        The number of games to play
    
    Returns
    -------
    The performance of the policy"""
    env = TictactoeEnv()
    opponent = OptimalPlayer(epsilon=epsilon)

    N_win, N_loss, N_draw = 0, 0, 0
    for iteration in range(N):
        env.reset()
        grid, end, _ = env.observe()

        if iteration < N // 2:
            policy_player = "X"
            opponent_player = "O"
        elif iteration >= N // 2:
            policy_player = "O"
            opponent_player = "X"
        
        opponent.set_player(opponent_player)
        if isinstance(policy, OptimalPlayer):
            policy.set_player(policy_player)

        while not end:
            
            if env.current_player == opponent.player:
                move = opponent.act(grid)
            else:
                move = policy.act(grid)

            grid, end, winner = env.step(move)
        
        if winner == policy_player:
            N_win += 1
        elif winner == opponent_player:
            N_loss += 1
        else:
            N_draw += 1
    
    return (N_win - N_loss) / N

def M_opt(policy, N=500):
    """Compute the performance of the policy against an optimal player"""
    return M(policy, 0, N)

def M_rand(policy, N=500):
    """Compute the performance of the policy against a random player"""
    return M(policy, 1, N)