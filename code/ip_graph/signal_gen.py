# import geomdl
# import itertools
# import copy
# import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# https: // www.datacamp.com/community/tutorials/networkx-python-graph-tutorial

import numpy as np
# rng = np.random.random_integers(12345)

workpath = "/home/tim/git/ip_graph/out/"
file_name = "{}curve.csv".format(workpath)

df = pd.read_csv(file_name, header=0, names=['x', 'y'])

pts = df[['x', 'y']].values
num_pts = pts.shape[0]
# time = np.linspace(num_pts)

plt.plot(pts[:, 0], pts[:, 1])
# plt.savefig("{}curve2.png".format(workpath))
# plt.close()

def arange2d(x0, x1, y0, y1, t0, eps = 0.6):
    n = np.int((((x1 - x0)**2 + (y1 - y0)**2) / eps)**0.5)
    if n < 6: 
        n = 6
    cx = np.vstack((np.linspace(x0, x1, n), np.linspace(y0, y1, n)))
    tx = np.linspace(t0, t0 + eps * n, n)
    return cx[:,1:], tx[1:]


trueground = np.array([[0.], [0.]])
timesteps = np.array([0.])
data = np.array([[0.], [0.]])

eps = 0.9

nsteps = pts.shape[0] - 1
# nsteps = 16

for i in range(nsteps):
    xx, tx = arange2d(pts[i][0], pts[i+1][0], pts[i][1], pts[i+1][1], timesteps[-1])
    trueground = np.append(trueground, xx, axis = 1)
    timesteps = np.append(timesteps, tx)

trueground = np.delete(trueground, 0, axis=1)
timesteps = np.delete(timesteps, -1)

noise = eps * np.random.randn(2, trueground.shape[1])
data = noise + trueground

# print(trueground, trueground.shape)
# plt.scatter(trueground[0, :], trueground[1, :])
plt.plot(data[0, :], data[1, :])
# plt.scatter(data[:100, 0], data[:100, 1])
plt.savefig("{}noise signal.png".format(workpath))
plt.close()

df = pd.DataFrame({
    'time': timesteps,
    'true_x': trueground[0, :],
    'true_y': trueground[1, :],
    'x': data[0, :],
    'y': data[1, :]
                   })

df.to_csv("{}noise.csv".format(workpath), index=False)
