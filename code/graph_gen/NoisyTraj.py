from scipy.signal import savgol_filter

import json
import os.path
import inspect
from networkx.readwrite.json_graph import jit_data, jit_graph
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_pydot import write_dot, read_dot
# nx.drawing.nx_pydot.write_dot
import networkx as nx

import geomdl
from geomdl import BSpline
from geomdl import utilities

from routes import Mgraph


# imitate the route of man walking through the building covering full area of corridors, 
# when we have multiple lines in the same corridor, we ca nperform matching of the data and 
# reconstruct the spatial map of location - magnetic 

class NoisyTraj():
    def __init__(self, tr):
        assert isinstance(tr, np.ndarray)
        self.curve = None

        if tr.shape[0] < 4:
            self.pts = None
            return

        curve = BSpline.Curve()
        curve.degree = 3
        # curve.delta = 0.01

        curve.ctrlpts = tr.tolist()
        curve.knotvector = utilities.generate_knot_vector(
            curve.degree, len(curve.ctrlpts))
        self.pts = np.array(curve.evalpts)
        self.curve = curve

    def transform_trajectory(self, eps, offsets, wlen=15, w_order=2):
        pts = np.copy(self.pts)

        # rot eff [-y x]
        diff = pts[3:-3, :] - pts[2:-4, :]
        # tangential vector
        # orthogonal noise
        rotdiff = np.array([-diff[:, 1], diff[:, 0]]).T
        off = offsets[np.random.randint(len(offsets))]
        pts[3:-3, :] = pts[3:-3, :] + off + eps * rotdiff * \
            np.random.randn(pts.shape[0] - 6, pts.shape[1])  # add noise

        xf = savgol_filter(pts[:, 0], wlen, w_order)
        yf = savgol_filter(pts[:, 1], wlen, w_order)
        return np.array([xf, yf])

    def data_gen(self, repeat = 6):
        obs = []
        # straight near walls
        dx = 0.25
        offsets = [[-dx, -dx], [-dx, dx], [dx, -dx], [dx, dx]]
        for i in range(4):
            obs.append(self.transform_trajectory(0.34, offsets, 9, 3))
        # noisy random
        eps = 1.5
        dx = 0.2
        offsets = [[-dx, -dx], [-dx, 0], [-dx, dx], [0, -dx],
                    [0, 0], [0, dx], [dx, -dx], [dx, 0], [dx, dx]]
        for i in range(abs(repeat - 4)):
            # dx = 0.15 / (1 + np.random.randint(3))  # rand step
            obs.append(self.transform_trajectory(eps, offsets, 13, 3))
        return obs


if __name__ == "__main__":
    mg = Mgraph(11)
    g, pos = mg.G, mg.pos

    n = 7
    selected = np.random.randint(0, len(mg.nodes), size=n)
    routes = np.vstack((selected[:-1], selected[1:]))
    datasetgt, trajs = mg.gen_routes(routes)

    fig = plt.figure(figsize=(3, 3), dpi=150)
    mg.plot_field(fig)

    noisy_dat = []

    for tr in datasetgt:
        nt = NoisyTraj(tr)
        if nt.curve == None:
            continue

        obs = nt.data_gen(repeat=7)
        noisy_dat.append(obs)
        for oi in obs:
            plt.plot(oi[0, :], oi[1, :], linewidth = 1)

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    # saveto = os.path.join(path, 'content/file.json')

    plt.tight_layout()
    saveto = os.path.join(path, 'content/noisy_traj.png')
    print("saveto:", saveto)
    plt.savefig(saveto)
    plt.show()
