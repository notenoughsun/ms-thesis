# import geomdl
# import itertools
# import copy
# import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
# https: // www.datacamp.com/community/tutorials/networkx-python-graph-tutorial

import numpy as np

# import networkx as nx

# from geomdl import exchange
# from geomdl import convert
# from geomdl import BSpline
# from geomdl.visualization import VisMPL
# from geomdl import utilities
# from geomdl import NURBS

cwd = os.getcwd()
savedir = os.path.join(cwd, "code/ip_graph/out")
# workpath = "/home/tim/git/ip_graph/out/"

def saveto(name): 
    return os.path.join(savedir, name)  

# file_name = "{}curve.csv".format(workpath)


class Bspline_curve_gen():
    staticmethod 
    def integrate(n_pts=50):
        data_shape = (2, n_pts)
        eps = 0.8
        dt = 0.05
        v0 = np.array([[0], [0]])

        noise = eps * np.random.randn(data_shape[0], data_shape[1] - 2)
        time = np.linspace(0, dt * n_pts, n_pts)

        acc = np.cumsum(noise, axis=0, dtype=float)
        fig, ax = plt.subplots(1, 3, sharex=True)
        speed = np.zeros(2, 1)
        speed = np.hstack(speed, dt * np.cumsum(acc, axis=0, dtype=float))
        
        traj =  dt * np.cumsum(speed, axis=0, dtype=float)
        ax[0].plot(time[:-2], acc[0, :], label = "acc")
        ax[0].plot(time[:-1], speed[0, :], label="speed")
        ax[0].plot(time[:-2], traj[0, :], label="traj")
        plt.legend()
        plt.savefig(saveto("Bspline_inetgrate.png"))

        plt.close()
        return traj

    def __init__(self):
        self.pts = Bspline_curve_gen.integrate()
    
    def plot(self, ax=None):
        # if not ax:
        #     ax = plt.gca()
        plt.plot(self.pts[:, 0], self.pts[:, 1])


bsgen = Bspline_curve_gen()
# fig, ax = plt.subplots()
bsgen.plot()
plt.savefig(saveto("Bspline_curve_gen.png"))
plt.show(block = True)


class Trajectory(object):
    def __init__(self, data = None):
        if not data:
            # set empty traj
            self.num_pts = 0
            # empty dataframe of some columns, xy reqired
            self.df = None
            self.pts = np.array([[], []])
        else:
            df = pd.read_csv(data, header=0, names=['x', 'y'])
            pts = df[['x', 'y']].values
            # num_pts = 
            self.num_pts = pts.shape[0]

            self.df = df
            self.pts = pts

    # def __init__(self) -> None:
        # super().__init__()

    def plot(self, ax = None):
        if not ax:
            ax = plt.gca()
        ax.plot(self.pts[:, 0], self.pts[:, 1])


# tr = Trajectory(os.path.join(cwd, "curve.csv"))


def more():
    # file_name = 
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
