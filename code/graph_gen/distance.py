import numpy as np
from scipy.spatial.distance import cdist, cosine
import matplotlib.pyplot as plt

# import json
import os.path
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

def cos_vec(u, v):
    dis = np.array([[cosine(ui, vi) for vi in v] for ui in u])
    return dis


def dist(u, v, c1 = 0.5, c2 = 0.5):
    d1 = cdist(u[:, :2], v[:, :2], 'euclidean')
    d2 = cos_vec(u[:, :2], v[:, 2:])
    return c1 * d1 + c2 * d2

n = 25
nc = np.array([
    np.linspace(0.01, 1, n),
    np.linspace(-1, 1, n),
    np.cos(np.linspace(0, 3, n)),
    np.linspace(-10, 1, n),
]).T

fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True,
                                    figsize=(12, 6))

z_cos = cos_vec(nc[:, 2:], nc[:, 2:])
z_dist = dist(nc, nc)
z2_dist = cdist(nc[:, :2], nc[:, :2], 'euclidean')


zmin, zmax = np.min([np.min(z_cos), np.min(z_dist), np.min(z2_dist)]), np.max([np.max(z_dist), np.max(z_cos), np.max(z2_dist)])
# x = np.linspace(0., 1., xn)
# y = np.linspace(0., 1., yn)

# xx, yy = np.meshgrid(nc[2], nc[3])
axes[0].imshow(z_cos, interpolation='bilinear', vmin=zmin, vmax=zmax,
                     extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
axes[0].set_title("cosine distance")
# xx, yy = np.meshgrid(nc[0], nc[1])
# axes[1].imshow(z_dist.T, interpolation='bilinear', resample=True)
im2 = axes[1].imshow(z_dist, interpolation='bilinear', vmin=zmin, vmax=zmax,
                     extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
axes[1].set_title("sum cos+euclidean distance")

axes[2].imshow(z2_dist, interpolation='bilinear', vmin=zmin, vmax=zmax,
                     extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
axes[2].set_title("euclidean distance")

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)

plt.savefig(os.path.join(path, 'content/distance fuc.png'))
plt.show()


