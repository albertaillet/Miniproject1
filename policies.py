import random

class EpsilonGreedy:
    """Epsilon-greedy policy
    
    Attributes
    ----------
    Q_table : QTable
        The Q-table to use to choose action when exploiting
    epsilon : float
        The epsilon value used to determine whether to explore or exploit
    player : str or int
        The player to use for the policy, either 'X' or 'O'

    Methods
    -------
    set_epsilon(epsilon)
        Set the epsilon value
    act(state)
        Choose an action given the state of the board
    """
    def __init__(self, Q_table, epsilon=0):
        self.epsilon = epsilon
        self.Q_table = Q_table
    
    def set_epsilon(self, epsilon):
        """Set the epsilon value"""
        self.epsilon = epsilon
    
    def act(self, state) -> int:
        """Choose an action given the state of the board"""
        Q_actions = self.Q_table[state]
        # Explore
        if random.random() < self.epsilon:
            actions = list(Q_actions.keys())
        # Exploit
        else: 
            actions = []
            max_val = max(Q_actions.values())
            for action in Q_actions:
                if Q_actions[action] == max_val:
                    actions.append(action)
        return random.choice(actions)


class EpsilonGreedyDecreasingExploration(EpsilonGreedy):
    """Epsilon-greedy policy with decreasing exploration rate

    Attributes
    ----------
    See EpsilonGreedy
    epsilon_min : float
        The minimum epsilon value to use int decreasing exploration formula
    epsilon_max : float
        The maximum epsilon value to use int decreasing exploration formula
    n_star : int
        The n* parameter to use in the decreasing exploration formula
    
    Methods
    -------
    See EpsilonGreedy
    update_epsilon(n)
        Update the epsilon value using the decreasing exploration formula depending on the step n
    """

    def __init__(self, Q_table, epsilon_min=0.1, epsilon_max=0.8, n_star=20000):
        super().__init__(Q_table)
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.n_star = n_star

    def update_epsilon(self, n):
        """Update the epsilon value using the decreasing exploration formula depending on the step n"""
        new_epsilon = max(self.epsilon_min, self.epsilon_max * (1 - (n / self.n_star)))
        self.set_epsilon(epsilon=new_epsilon)