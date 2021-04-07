from geomdl import BSpline
import geomdl
from geomdl import exchange
from geomdl import convert

import pandas as pd

# Create a BSpline surface instance (Bezier surface)

surf = BSpline.Surface()

# Set degrees
surf.degree_u = 3
surf.degree_v = 2

# Set control points
control_points = [[0, 0, 0], [0, 4, 0], [0, 8, -3],
                [2, 0, 6], [2, 4, 0], [2, 8, 0],
                [4, 0, 0], [4, 4, 0], [4, 8, 3],
                [6, 0, 0], [6, 4, -3], [6, 8, 0]]
surf.set_ctrlpts(control_points, 4, 3)

# Set knot vectors
surf.knotvector_u = [0, 0, 0, 0, 1, 1, 1, 1]
surf.knotvector_v = [0, 0, 0, 1, 1, 1]

# Set evaluation delta (control the number of surface points)
surf.delta = 0.05

# Get surface points (the surface will be automatically evaluated)
surface_points = surf.evalpts
# print(surface_points)

workpath = "/home/tim/git/perception/code/surface/"
file_name = "{}surface.csv".format(workpath)

geomdl.exchange.export_csv(surf, file_name=file_name, point_type='evalpts')

# s = np.linspace(0, 10, 240)
# t = np.linspace(0, 10, 240)
# xGrid, yGrid = np.meshgrid(s, t)
# z = np.random.randn(s.shape[0], t.shape[0]) * np.cos(sGrid) * np.sin(tGrid)

# surface = go.Surface(x=x, y=y, z=z)

surface_points2 = pd.read_csv(file_name, header=0, names=['x', 'y', 'z'])
# print()
pts = surface_points2[['x', 'y', 'z']].values
# plt.plot(pts[:, 0], pts[:, 1])
