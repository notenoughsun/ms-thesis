#%%
from zspace import Zspace
boxsize = 11
zs = Zspace(boxsize)
fig, ax = zs.plot_3surf()
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import animation as anim
from progress.bar import FillingCirclesBar

from process import Process

boxsize = 11
zs = Zspace(boxsize)

sim = Process(zs=zs)
sim.params.n_routes = 26

sim.generate_input_data()
sim.plot_data(alpha=0.6)

data = sim.df[:5]
sim.run(data)
# dataset =
sim.mg.generate_routes(26)

#%%
# run single
fig = plt.figure(figsize=(3, 3), dpi=150)
plt.axis('off')
sim.zs.plot_field(fig)


data = sim.df[:1]
Process.plot_traj(data['traj'][0])
sim.run(data, fig=fig)

plt.tight_layout()
plt.axis('off')
# saveto = os.path.join(sim.path, 'content/sim_test.png')
# plt.savefig(saveto)
# plt.show()

# %%
# experiment, change params
# n = 7
# betas =
# alphas =
fig = plt.figure(figsize=(3, 3), dpi=150)
plt.axis('off')
sim.zs.plot_field(fig)

alphas = np.array([0.05, 0.001, 0.05, 0.01])
betas = np.array([0.05, 0.001, 0.1])

oi = data[['oi']]
oi
#%%
n = 4
scale = [np.linspace(1, 4, n), np.ones(n)]

# sim.params.show_cov = True

for i in range(n):
    sim.params.alphas = alphas * scale[0, i]
    sim.params.betas = betas * scale[1, i]
    #  = alphas
    #  = betas
    sim.run(data, fig=fig)
  
plt.show()
#%%
matrice = sim.zs.Z.T[::-1, :]
print(matrice)
data = sim.df[:1]
traj = data['traj'][0]
# traj += np.ones((2, 1)) * 1

fig = plt.figure(figsize=(3, 3), dpi=150)
plt.axis('off')
sim.zs.plot_field(fig)
Process.plot_traj(traj)
# coords = data[['oi']][0]


def plot_cond(traj):
    # condition x on maze occupancy grid data
    c = np.array(traj + [[0.5], [0.5]]).astype(int)
    sel = sim.zs.Z[c[0], c[1]] > 0

    Process.plot_traj(traj)
    pts_sc = traj[:, sel]
    plt.scatter(pts_sc[0, :], pts_sc[1, :], color='r')
    pts_sc = traj[:, np.logical_not(sel)]
    plt.scatter(pts_sc[0, :], pts_sc[1, :], color='b')

    return sel

data = sim.df[:5]

sim.zs.plot_field(fig)
plt.axis('off')

# for i in range(5):
traj = data['traj'][0]
plot_cond(traj)
traj += np.ones((2, 1))
plot_cond(traj)

# %%
pts = np.random.multivariate_normal(np.zeros(2), 1e-12 * np.eye(2, 2), 100)


sel = np.bitwise_and(pts[:, 0] > 0, pts[:, 1] > 0)
pts_sc = pts[sel]
plt.scatter(pts_sc[:, 0], pts_sc[:, 1], color='r')
pts_sc = pts[np.logical_not(sel)]
plt.scatter(pts_sc[:, 0], pts_sc[:, 1], color='b')

# %%
