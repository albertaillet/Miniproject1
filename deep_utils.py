import string
import torch
import random
import numpy as np
from collections import namedtuple, deque
from tic_plot import plot_grid
from typing import Union
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'player'))
p2v = {'X': 1, 'O': -1}


class ReplayBuffer(object):
    """
    A class for the Replay Buffer
    uses code from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory

    Attributes
    ----------
    buffer : deque
        The buffer
    batch_size : int
        The batch size used when sampling

    Methods
    -------
    push(state, action, next_state, reward, player)
        Save a transition to the buffer
    get_batch(batch_size=None)
        Get a batch of transitions from the buffer
    __len__()
        Get the length of the buffer
    has_one_batch(batch_size=None)
        Check if the buffer has at least one batch
    """

    def __init__(self, buffer_size: int, batch_size: int) -> None:
        self.buffer = deque([], maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, *args) -> None:
        """Save a transition to the buffer
        
        Parameters
        ----------
        *args : Transition
            Transition agruments: state, action, next_state, reward, player
        """
        self.buffer.append(Transition(*args))

    def get_batch(self, batch_size=None) -> None:
        """Get a batch of transitions from the buffer

        Parameters
        ----------
        batch_size : int, optional
            The batch size used when sampling, (the default is None, which uses the buffer's batch size)
        
        Returns
        -------
        list of Transitions
        """
        if batch_size is None:
          batch_size = self.batch_size
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Get the length of the buffer"""
        return len(self.buffer)

    def has_one_batch(self, batch_size=None) -> bool:
        """Check if the buffer has at least one batch"""
        if batch_size is None:
          batch_size = self.batch_size
        return len(self) >= batch_size

def state_to_tensor(state: np.ndarray, player: int) -> torch.Tensor:
    """Convert a the state represntation of the board the corresponding tensor

    Parameters
    ----------
    state : np.ndarray
        The state
    player : str or int
        The player
    
    Returns
    -------
    torch.Tensor
        The tensor representation of the state
    """
    if player==-1:
        opponent_player = 1
    elif player==1:
        opponent_player = -1
    elif p2v[player]==-1:
        player = p2v[player]
        opponent_player = 1
    elif p2v[player]==1:
        player = p2v[player]
        opponent_player = -1
    else:
        raise ValueError(f"Player should be 1 or -1, player={player}")

    t = np.zeros((3, 3, 2), dtype=np.float32)
    t[:, :, 0] = (state == player)
    t[:, :, 1] = (state == opponent_player)
    return torch.tensor(t, dtype=torch.float32)


# Policies

class DeepEpsilonGreedy:
    """Epsilon-greedy policy
    
    Attributes
    ----------
    net : torch.nn.Module
        The neural network to use to choose action when exploiting
    epsilon : float
        The epsilon value used to determine whether to explore or exploit
    player : str or int
        The player to use for the policy, either 'X' or 'O'

    Methods
    -------
    set_epsilon(epsilon)
        Set the epsilon value
    set_player(player)
        Set the player to use for the policy
    act(state)
        Choose an action given the state of the board
    """

    def __init__(self, 
                 net: torch.nn.Module, 
                 epsilon: float=0, 
                 player: str='X') -> None:
        self.net = net
        self.epsilon = epsilon
        self.player = player
    
    def set_epsilon(self, epsilon: float) -> None:
        """Set the epsilon value"""
        self.epsilon = epsilon

    def set_player(self, player: str) -> None:
        """Set the player to use for the policy"""
        self.player = player
    
    def act(self, state: np.ndarray) -> int:
        """Choose an action given the state of the board"""
        # Explore
        if random.random() < self.epsilon:
            available = np.nonzero(state.flatten() == 0)
            return int(random.choice(available[0]))
        # Exploit
        else:
            state = state_to_tensor(state, self.player)
            with torch.no_grad():
                return torch.argmax(self.net(state)).item()
            


class DeepEpsilonGreedyDecreasingExploration(DeepEpsilonGreedy):
    """Epsilon-greedy policy with decreasing exploration rate

    Attributes
    ----------
    See DeepEpsilonGreedy
    epsilon_min : float
        The minimum epsilon value to use int decreasing exploration formula
    epsilon_max : float
        The maximum epsilon value to use int decreasing exploration formula
    n_star : int
        The n* parameter to use in the decreasing exploration formula
    
    Methods
    -------
    See DeepEpsilonGreedy
    update_epsilon(n)
        Update the epsilon value using the decreasing exploration formula depending on the step n
    """

    def __init__(self, 
                 net: torch.nn.Module, 
                 player: str='X', 
                 epsilon_min: float= 0.1, 
                 epsilon_max: float=0.8, 
                 n_star: int=20000) -> None:
        super().__init__(net, player=player)
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.n_star = n_star

    def update_epsilon(self, n: int) -> None:
        """Update the epsilon value using the decreasing exploration formula depending on the step n"""
        new_epsilon = max(self.epsilon_min, self.epsilon_max * (1 - (n / self.n_star)))
        self.set_epsilon(epsilon=new_epsilon)


# Debug 

def examples_output_images(model: torch.nn.Module) -> np.ndarray:
    """Generate examples of the model's output
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to produce the output images from
        
    Returns
    -------
    Image of the model's outputs for the three specified examples
    """

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
    
def debug_table(d: dict) -> str:
    """Print a table of the given dictionary

    Parameters
    ----------
    d : dict
        The nested dictionary to print, should have the following structure:
        {'X': {'win': 0, 'draw': 0, 'loss': 0}, 
         'O': {'win': 0, 'draw': 0, 'loss': 0}
    
    Returns
    -------
    Formatted table of the given dictionary
    """
    return '  \n'.join(['|||||', '|-|-|-|-|']+['|'.join(['', f'player: {p}']+[f"{n}: {o}" for n, o in m.items()]+['']) for p, m in d.items()])