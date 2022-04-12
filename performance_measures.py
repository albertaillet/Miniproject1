from tic_env import TictactoeEnv, OptimalPlayer

"""
M_opt measures the performance of pi against the optimal policy. 
To compute M_opt, we run pi against Opt(0) for N = 500 games for different random seeds. 
pi makes the 1st move in 250 games, and Opt(0) makes the 1st move in the rest. 
We count how many games pi wins (N_win) and loses (N_loss) and define M_opt = N_win - N_loss / N.

M_rand measures the performance of against the random policy. 
To compute M_rand, we repeat what we did for computing M_opt but by using Opt(1) instead of Opt(0).
"""

def M(policy, epsilon, N=100):
    env = TictactoeEnv()
    
    N_win, N_loss = 0, 0
    for i in range(N):
        env.reset()
        
        policy_player = "X"
        player = "O"
        if i < 250:
            player, policy_player = policy_player, player
        
        opt_player = OptimalPlayer(epsilon=epsilon, player=player)

        while not env.end:
            
            if env.current_player == opt_player.player:
                move = opt_player.act(grid)
            else:
                move = policy(grid)

            grid, _, _ = env.step(move, print_grid=False)
        
        if env.winner == policy_player:
            N_win += 1
        else:
            N_loss += 1
    
    assert N_win + N_loss == N

    return (N_win - N_loss) / N

def M_opt(policy, N=100):
    return M(policy, 1, N)

def M_rand(policy, N=100):
    return M(policy, 0, N)
