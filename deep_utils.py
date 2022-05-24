import torch
import random
import numpy as np
from collections import namedtuple, deque
from tic_plot import plot_grid
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'player'))
p2v = {'X': 1, 'O': -1}

# Replay buffer
# uses code from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory

class ReplayBuffer(object):

    def __init__(self, buffer_size, batch_size):
        self.buffer = deque([], maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def get_batch(self, batch_size=None):
        if batch_size is None:
          batch_size = self.batch_size
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def has_one_batch(self, batch_size=None):
        if batch_size is None:
          batch_size = self.batch_size
        return len(self) >= batch_size

# state to tensor
# def state_to_tensor(state, player):
#     if player==-1:
#         opponent_player = 1
#     elif player==1:
#         opponent_player = -1
#     elif p2v[player]==-1:
#         player = p2v[player]
#         opponent_player = 1
#     elif p2v[player]==1:
#         player = p2v[player]
#         opponent_player = -1
#     else:
#         raise ValueError(f"Player should be 1 or -1, player={player}")

#     t = np.zeros((3, 3, 2), dtype=np.float32)
#     t[:, :, 0] = (state == player)
#     t[:, :, 1] = (state == opponent_player)
#     return torch.tensor(t, dtype=torch.float32)

def state_to_tensor(state, _):
  return torch.tensor(state, dtype=torch.float32)


# Policies

class DeepEpsilonGreedy:
    def __init__(self, net, epsilon=0, n_actions=9, player='X'):
        self.net = net
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.player = player
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_player(self, player):
        self.player = player
    
    def act(self, state):
      if random.random() > self.epsilon:
          state = state_to_tensor(state, self.player)
          with torch.no_grad():
              return torch.argmax(self.net(state)).item()
      else:
          #return random.randrange(self.n_actions)
          available = np.nonzero(state.flatten() == 0)
          return int(random.choice(available[0]))


class DeepEpsilonGreedyDecreasingExploration(DeepEpsilonGreedy):

    def __init__(self, net, n_actions=9, player='X', epsilon_min= 0.1, epsilon_max=0.8, n_star=20000):
        super().__init__(net, n_actions=n_actions, player=player)
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.n_star = n_star

    def update_epsilon(self, n):
        new_epsilon = max(self.epsilon_min, self.epsilon_max * (1 - (n / self.n_star)))
        self.set_epsilon(epsilon=new_epsilon)


# Debug 

def examples_output_images(model):
    examples = [
                ((0, 0, 0, 0, 0, 0, 0, 0, 0), 1),
                ((0, -1, -1, 0, 1, 1, 0, 0, 0), 1),
                ((1, -1, -1, 1, 1, 0, 0, 0, 0), -1),             
    ]
    imgs = []
    for state, player in examples:
        a = np.array(state).reshape((3, 3))
        t = state_to_tensor(a, player)
        with torch.no_grad():
            out = model(t).cpu().numpy().reshape((3, 3))
        plot_grid(a, out, clim=(-2,2))
        fig = plt.gcf()
        plt.close()
        # redraw the canvas
        fig.canvas.draw()
        # convert canvas to image using numpy
        imgs.append(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8))
    imgs = np.array(imgs)
    W, H = fig.canvas.get_width_height()[::-1]
    N = len(examples)
    img = imgs.reshape((N*W, H, 3))
    return img
    
def debug_table(d):
  return '  \n'.join(['|||||', '|-|-|-|-|']+['|'.join(['', f'player: {p}']+[f"{n}: {o}" for n, o in m.items()]+['']) for p, m in d.items()])