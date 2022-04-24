import numpy as np
import matplotlib.pyplot as plt

def plot_grid(grid=None, heatmap=None, cmap='jet'):
    lim = (-0.5, 2.5)
    msize = 2000
    plt.gca().set_aspect('equal')
    plt.xlim((-0.5, 2.5))
    plt.ylim((2.5, -0.5))

    if heatmap is not None:
        plt.imshow(heatmap, interpolation='none', aspect='equal', cmap=cmap)
    if grid is not None:
        if 1 in grid:
            plt.scatter(*np.nonzero(grid.T==1), marker="x", s=msize, c='k')
        if -1 in grid:
            plt.scatter(*np.nonzero(grid.T==-1), marker="o", s=msize, edgecolors='k', facecolors='none')
        
    plt.xticks(np.arange(0.5, 2.5))
    plt.yticks(np.arange(0.49, 2.5))
    plt.tick_params(left = False, 
                    right = False, 
                    labelleft = False,
                    labelbottom = False, 
                    bottom = False)
    plt.grid()

def render_grid(grid):
    value2player = {0: '-', 1: 'X', -1: 'O'}
    for i in range(3):
        print('|', end='')
        for j in range(3):
            print(value2player[int(state[i,j])], end=' ' if j<2 else '')
        print('|')
    print()


if __name__ == "__main__":
    state = np.zeros((3, 3))
    state[1, 2] = 1
    state[2, 1] = -1
    render_grid(state)
    plot_grid(state, state)
    plt.show()

    state = np.array([[ 1, 1, 0],
                    [ 0, -1, 0],
                    [-1, 0, 1]])
    render_grid(state)
    plot_grid(state)
    plt.show()

    state = np.array([0, 0, 1, 0, -1, -1, 0, 0, 0]).reshape((3, 3))
    render_grid(state)
    plot_grid(state, state, cmap='plasma')
    plt.show()

    state = np.array([0, 0, -1, 0, 1, 1, 0, 0, 0]).reshape((3, 3))
    render_grid(state)
    plot_grid(None, state)
    plt.show()