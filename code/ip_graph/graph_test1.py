from geomdl import exchange
from geomdl import convert
from geomdl import BSpline
from geomdl.visualization import VisMPL
from geomdl import utilities
from geomdl import NURBS
import geomdl
import itertools
import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# https: // www.datacamp.com/community/tutorials/networkx-python-graph-tutorial

import numpy as np
# rng = np.random.random_integers(12345)

workpath = "/home/tim/git/ms-thesis/code/ip_graph/out"

class Location():
    def __init__(self, n_routers = 6, sizex = 10, sizey = 10):
        
    # def generate_location():
        self.n_routers = n_routers
        x = np.random.randint(low=0, high=sizex, size=n_routers)
        y = np.random.randint(low=0, high=sizey, size=n_routers)
        router_coord_ids = range(n_routers)
        self.df = pd.DataFrame({'name': router_coord_ids,
                        'x': x,
                        'y': y})

    def save(self, path = workpath):                    
        self.df.to_csv("{}out.csv".format(path), index=False)

    def show(self, path=workpath):
        # figure, ax = plt.figure()
        plt.scatter(self.df['x'].values, self.df['y'].values)
        plt.show()
        plt.savefig("{}img.png".format(path))


# loc = Location(n_routers=12, sizex=100, sizey=100)
# loc.save()
# loc.show()

# class Trajectory():
#     def __init__(self, targets):
#         self.position = [0, 0]
#         self.targets = targets
#         self.ntargets = targets.shape[0]
#         self.step_size = 0.2
#         self.tgt = targets[:, 0]

#     def random_walk(self):
#         pos = self.position
#         print(self.targets.shape, self.tgt)
#         id = np.random.randint(self.ntargets)
#         self.tgt = self.targets[:, 0]
#         pos = 


# tr = Trajectory(targets = loc.df["x, y"].values)
# ctrlpts = loc.df["x, y"].values
# print("ctrlpts", ctrlpts)
sizex = sizey = 10
n_routers = 16
n  = n_routers
num_ctrlpts = n_routers


# # Create a 3-dimensional B-spline Curve
# curve = NURBS.Curve()
# # Set degree
# curve.degree = 2
# degree = 2
# m = n + curve.degree + 1
# # Set control points (weights vector will be 1 by default)
# # Use curve.ctrlptsw is if you are using homogeneous points as Pw
# # curve.ctrlpts = [[10, 5, 10], [10, 20, -30], [40, 10, 25], [-10, 5, 0]]
# curve.ctrlpts = ctrlpts
# # Set knot vector
# curve.knotvector = geomdl.knotvector.generate(degree, num_ctrlpts)
# list(np.array([[0] * n, [1] * n]).flatten())
# curve.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]

# # Set evaluation delta (controls the number of curve points)
# curve.delta = 0.05

# # Get curve points (the curve will be automatically evaluated)
# curve_points = curve.evalpts

# Create a B-Spline curve
curve = BSpline.Curve()

# Set up the curve
curve.degree = 4
# curve.ctrlpts = [[5.0, 10.0], [15.0, 25.0], [30.0, 30.0], [45.0, 5.0], [
#     55.0, 5.0], [70.0, 40.0], [60.0, 60.0], [35.0, 60.0], [20.0, 40.0]]


# x = np.random.randint(low=0, high=sizex, size=n_routers)
# y = np.random.randint(low=0, high=sizey, size=n_routers)

# ctrlpts = list(zip(x, y))

curve.ctrlpts = [[5.0, 10.0], [15.0, 25.0], [30.0, 30.0], [45.0, 5.0], [
    55.0, 5.0], [70.0, 40.0], [60.0, 60.0], [35.0, 60.0], [20.0, 40.0]]


# Auto-generate knot vector
curve.knotvector = utilities.generate_knot_vector(
    curve.degree, len(curve.ctrlpts))

# Set evaluation delta
curve.delta = 0.01

# Plot the control point polygon and the evaluated curve
curve.vis = VisMPL.VisCurve2D()
# https://nurbs-python.readthedocs.io/en/5.x/module_vis_mpl.html
curve.render()
plt.savefig("{}curve.png".format(workpath))
plt.close()

# Import convert module

# BSpline to NURBS
crv_rat = convert.bspline_to_nurbs(curve)

file_name = "{}curve.csv".format(workpath)
# exchange.export_csv(curve, file_name=file_name, point_type='evalpts')

geomdl.exchange.export_csv(curve, file_name=file_name, point_type='evalpts')

curvepts = pd.read_csv(file_name, header = 0, names = ['x', 'y'])
# print()
pts = curvepts[['x', 'y']].values
plt.plot(pts[:, 0], pts[:, 1])
# plt.show()
plt.savefig("{}curve2.png".format(workpath))
plt.close()
