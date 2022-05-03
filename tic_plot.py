import numpy as np
import matplotlib.pyplot as plt

def plot_grid(grid=None, heatmap=None, cmap='jet', clim=(-1,1)):
    msize = 2000
    plt.gca().set_aspect('equal')
    plt.xlim((-0.5, 2.5))
    plt.ylim((2.5, -0.5))

    if grid is not None:
        if 1 in grid:
            plt.scatter(*np.nonzero(grid.T==1), marker="x", s=msize, c='k')
        if -1 in grid:
            plt.scatter(*np.nonzero(grid.T==-1), marker="o", s=msize, edgecolors='k', facecolors='none')
    if heatmap is not None:
        for (i, j), value in np.ndenumerate(heatmap.T):
            if not np.isnan(value):
                plt.text(i-0.2, j+0.05, '{:.4f}'.format(value))
        plt.imshow(heatmap, interpolation='none', aspect='equal', cmap=cmap)
        plt.colorbar()
        plt.clim(clim)
        
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
    map = np.empty((3, 3))
    map[:] = np.NaN
    map[state==0] = 0.123456789
    map[1, 0] = -0.3456789
    map[0, 1] = 0.56789
    render_grid(state)
    plot_grid(state, map, cmap='jet')
    plt.clim(-1, 1)
    plt.show()

    state = np.array([0, 0, -1, 0, 1, 1, 0, 0, 0]).reshape((3, 3))
    render_grid(state)
    plot_grid(None, state)
    plt.show()
# %%
