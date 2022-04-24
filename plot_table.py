# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

# %%
state = np.zeros((3, 3))
state[1, 2] = 1
state[2, 1] = -1

# %%
state = np.array([[ 1, 1, 0],
                  [ 0, -1, 0],
                  [-1, 0, 1]])

# %%
state = np.array([0, 0, 1, 0, -1, -1, 0, 0, 0]).reshape((3, 3))

# %%
state = np.array([0, 0, -1, 0, 1, 1, 0, 0, 0]).reshape((3, 3))

lim = (-0.5, 2.5)
msize = 2000
#plt.imshow(state)
plt.xticks(np.arange(-0.5, 3.5))
plt.yticks(np.arange(-0.5, 3.5))
if 1 in state:
    plt.scatter(*np.nonzero(state.T==1), marker="x", s=msize, c='k')
if -1 in state:
    plt.scatter(*np.nonzero(state.T==-1), marker="o", s=msize, edgecolors='k', facecolors='none')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.xlim((-0.5, 2.5))
plt.ylim((2.5, -0.5))
plt.gca().set_aspect('equal')
plt.grid()

# %%
print(state)

#%%
value2player = {0: '-', 1: 'X', -1: 'O'}
for i in range(3):
    print('|', end='')
    for j in range(3):
        print(value2player[int(state[i,j])], end=' ' if j<2 else '')
    print('|')
print()