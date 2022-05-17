import random

class EpsilonGreedy:
    def __init__(self, Q_table, epsilon=0):
        self.epsilon = epsilon
        self.Q_table = Q_table
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
    
    def act(self, state):
        Q_actions = self.Q_table[state]
        # explore
        if random.random() < self.epsilon:
            actions = list(Q_actions.keys())
        # exploit
        else: 
            actions = []
            max_val = max(Q_actions.values())
            for action in Q_actions:
                if Q_actions[action] == max_val:
                    actions.append(action)
        return random.choice(actions)


class EpsilonGreedyDecreasingExploration(EpsilonGreedy):

    def __init__(self, Q_table, epsilon_min=0.1, epsilon_max=0.8, n_star=20000):
        super().__init__(Q_table)
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.n_star = n_star

    def update_epsilon(self, n):
        new_epsilon = max(self.epsilon_min, self.epsilon_max * (1 - (n / self.n_star)))
        self.set_epsilon(epsilon=new_epsilon)