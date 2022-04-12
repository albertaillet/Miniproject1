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
    env = TictactoeEnv()
    opponent = OptimalPlayer(epsilon=epsilon)

    N_win, N_loss, N_draw = 0, 0, 0
    for iteration in range(N):
        env.reset()
        grid, _, __ = env.observe()
      
        policy_player = "O"
        opponent_player = "X"
        if iteration >= (N // 2):
            opponent_player, policy_player = policy_player, opponent_player
        
        policy.player = policy_player
        opponent.player = opponent_player

        end = False
        while not end:
            
            if env.current_player == opponent.player:
                move = opponent.act(grid)
            else:
                move = policy.act(grid)

            grid, end, winner = env.step(move, print_grid=False)
        
        if winner == policy_player:
            N_win += 1
        elif winner == opponent_player:
            N_loss += 1
        else:
            N_draw += 1
    
    return (N_win - N_loss) / N

def M_opt(policy, N=500):
    return M(policy, 0, N)

def M_rand(policy, N=500):
    return M(policy, 1, N)
