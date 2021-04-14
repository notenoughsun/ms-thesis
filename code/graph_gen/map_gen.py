from scipy.interpolate import interpolate
# from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from routes import Mgraph

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt

# add random noise on walls edges
W = nx.Graph()

for i in range(boxsize - 1):
  for j in range(boxsize - 1):
    if Z[i][j] == 0:
      W.add_node(xytoname([i, j]), pos = [i, j])
      if i > 0 and Z[i - 1][j] == 0:
        W.add_edge(xytoname([i, j]), xytoname([i - 1, j]), weight = 1.)
      if j > 0 and Z[i][j - 1] == 0:
        W.add_edge(xytoname([i, j]), xytoname([i, j - 1]), weight = 1.)

posw = dict(W.nodes.data("pos"))
# nx.draw(W, pos=posw, alpha=0.9, node_size=15, width=2)



x = np.linspace(-0.5, boxsize - 0.5, boxsize - 1)
y = np.linspace(-0.5, boxsize - 0.5, boxsize - 1)

xx, yy = np.meshgrid(x, y)
z = np.random.randn(boxsize - 1, boxsize - 1)

interp = interpolate.interp2d(xx, yy, z, kind='linear')
zz = interp(x, y)

fig, axs = plt.subplots(nrows=1, ncols=1, subplot_kw={'xticks': [], 'yticks': []})
axs.set_aspect('equal')
# fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi = 150, subplot_kw={'xticks': [], 'yticks': []})
axs.imshow(zz, interpolation='spline36', resample=True)

plt.axis("equal")

x = np.arange(-0.5, boxsize - 0.5, 1)
y = np.arange(-0.5, boxsize - 0.5, 1)

nx.draw(G, pos=pos, alpha=0.3, node_size=5, width=1, edge_color='w', node_color='w')

for i in range(boxsize - 1):
  for j in range(boxsize - 1):
    if Z[i][j] == 0:
      rect = patches.Rectangle(xy = (i - 0.5, j - 0.5), width = 1., height=1., color = 'w')
      axs.add_patch(rect)

# nx.draw(W, pos=posw, alpha=0.9, edge_color='k', width=8, node_size=40, node_color='k')

for pts in ds[:10]:
  for xf, yf in pts:
    plt.plot(xf, yf, label = "savgol_filter", linewidth = 2) 

plt.show()