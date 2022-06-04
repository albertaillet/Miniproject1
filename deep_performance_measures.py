from tic_env import TictactoeEnv, OptimalPlayer

"""
M_opt measures the performance of pi against the optimal policy. 
To compute M_opt, we run pi against Opt(0) for N = 500 games for different random seeds. 
pi makes the 1st move in 250 games, and Opt(0) makes the 1st move in the rest. 
We count how many games pi wins (N_win) and loses (N_loss) and define M_opt = N_win - N_loss / N.

M_rand measures the performance of against the random policy. 
To compute M_rand, we repeat what we did for computing M_opt but by using Opt(1) instead of Opt(0).
"""

def M(policy, epsilon, N, debug):
    """Compute the performance of the policy against the an optimal player with a given epsilon
    
    Parameters
    ----------
    policy : DeepEpsilonGreedy
        The policy to compute the performance of, should have an act and set_player method
    epsilon : float
        The epsilon value to use for the optimal player
    N : int
        The number of games to play
    debug : bool
        Whether to return the game statistics
    
    Returns
    -------
    if debug:
        The performance of the policy and a dictionary with the number of wins, losses, and draws for each player
        with format: {'X': {'win': 0, 'draw': 0, 'loss': 0}, 
                      'O': {'win': 0, 'draw': 0, 'loss': 0}}
    else:
        The performance of the policy"""
    env = TictactoeEnv()
    opponent = OptimalPlayer(epsilon=epsilon)
    original_policy_player = policy.player

    N_win, N_loss, N_draw = 0, 0, 0
    games = {'X': {'win': 0, 'draw': 0, 'loss': 0}, 
             'O': {'win': 0, 'draw': 0, 'loss': 0}}
    for iteration in range(N):
        env.reset()
        grid, end, _ = env.observe()

        if 0 <= iteration < N // 2:
            policy_player = "X"
            opponent_player = "O"
        elif N // 2 <= iteration < N:
            policy_player = "O"
            opponent_player = "X"
        else: 
            raise ValueError(f"iteration must be less than N, itr: {iteration}")
        
        opponent.set_player(opponent_player)
        policy.set_player(policy_player)

        while not end:
            if env.current_player == opponent_player:
                move = opponent.act(grid)
            else:
                move = policy.act(grid)
            
            valid_move = env.check_valid(move)
            if not valid_move and env.current_player == policy_player:
                winner = opponent_player
                end = True
            elif valid_move: 
                grid, end, winner = env.step(move)   
            else:
                raise ValueError("Optimal player took an invalid move")
        
        if winner == policy_player:
            N_win += 1
            games[policy_player]['win'] += 1
        elif winner == opponent_player:
            N_loss += 1
            games[policy_player]['loss'] += 1
        elif winner == None:
            N_draw += 1
            games[policy_player]['draw'] += 1
        else:
            raise ValueError(f"winner must be None, or X or O, winner: {winner}")


    policy.set_player(original_policy_player)
    assert N_win + N_loss + N_draw == N
    if debug:
        return (N_win - N_loss) / N, games
    return (N_win - N_loss) / N

def M_opt(policy, N=500, debug=False):
    """Compute the performance of the policy against an optimal player"""
    return M(policy, 0, N, debug)

def M_rand(policy, N=500, debug=False):
    """Compute the performance of the policy against a random player"""
    return M(policy, 1, N, debug)