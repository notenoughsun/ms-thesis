#%%
from typing import Tuple
from numpy.core.numeric import indices
# from scipy.interpolate import interpolate
import scipy.interpolate as interpolate
# from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt

from routes import Mgraph
from NoisyTraj import NoisyTraj
# from code.graph_gen.routes import Mgraph
# from code.graph_gen.NoisyTraj import NoisyTraj

from tools.objects import FilterTrajectory, Gaussian
from tools.plot import plot2dcov
from filters.pf import PF


import json
import os.path
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import animation as anim
from progress.bar import FillingCirclesBar

from scipy.stats import norm as gaussian


class Zspace(Mgraph):
  '''
  3d surface of observations
  '''
  def __init__(self, boxsize = 9):
    super().__init__(boxsize = boxsize)
    extent = (0.5, boxsize - 1.5, 0.5, boxsize - 1.5)

    self.extent = extent
    x = np.linspace(extent[0], extent[1], boxsize - 1)
    y = np.linspace(extent[0], extent[1], boxsize - 1)

    # np.linspace()

    xx, yy = np.meshgrid(x, y)

    zz = []
    self.z_true = []

    self._mapQ = np.diag(np.ones(3) * 1e-2)

    for i in range(3):
      if i == 2:
        z = np.random.randn(boxsize - 1, boxsize - 1) * 5e-1
      else:
        z = np.random.randn(boxsize - 1, boxsize - 1)
      zfun_smooth_rbf = interpolate.Rbf(xx, yy, z, function='cubic', smooth=0.5)

      interp = interpolate.interp2d(xx, yy, z, kind='linear')

      # zfun_smooth_rbf = interp.Rbf(x_sparse, y_sparse, z_sparse_smooth, function='cubic', smooth=0)  # default smooth=0 for interpolation
      # z_dense_smooth_rbf = zfun_smooth_rbf(x_dense, y_dense)  # not really a function, but a callable class instance
      
      self.z_true.append(zfun_smooth_rbf)
      zz.append(interp(x, y))

    self.zz = zz
    # pass

  # def obs(self, state):
  #   '''
  #   state = [x, y, theta]
  #   return observations in global frame
  #   '''
  #   z = [self.z_true[i](state[0], state[1]) for i in range(3)]
  #   return z

  def plot_walls(self, ax):
    if ax == None:
      fig = plt.figure(figsize=(3,3), dpi = 150)
      ax = plt.gca()

    for i in range(self.boxsize):
      for j in range(self.boxsize):
        if self.Z[i][j] == 0:
          rect = patches.Rectangle(xy = (i - 0.5, j - 0.5), width = 1., height=1., color = 'gray')
          ax.add_patch(rect)

  def plot_surf(self, id = 0, fig = None):
    if fig == None:
      fig = plt.figure(figsize=(3,3), dpi = 150)
    ax = plt.gca()

    # fig, axs = plt.subplots(nrows=1, ncols=1, subplot_kw={'xticks': [], 'yticks': []})
    # axs.set_aspect('equal')
    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi = 150, subplot_kw={'xticks': [], 'yticks': []})
    ax.imshow(self.zz[0], interpolation='spline36', resample=True, extent = self.extent)

    # nx.draw(self.G, pos=self.pos, alpha=0.3, node_size=5, width=1, edge_color='w', node_color='w')

    # plt.scatter(xx, yy, color='r')
    self.plot_walls(ax)

    plt.axis("equal")
    self.draw_graph()

  def plot_3surf(self, fig = None, ax= None, show_colorbar = True):
    if fig == None:
      fig, ax = plt.subplots(1, 3, figsize=(10, 3), dpi = 150)
    indices =['x', 'y', 'z']
    # ax.axis('off')
    [axi.set_axis_off() for axi in ax.ravel()]
    # plt.axis('off')
    zmin, zmax = np.min(self.zz), np.max(self.zz)

    for k in range(3):
      im = ax[k].imshow(self.zz[k], interpolation='spline36', \
        resample=True, extent = self.extent,\
        vmin=zmin, vmax=zmax, aspect='auto', cmap='viridis')
      ax[k].set_title(indices[k])
      for i in range(self.boxsize):
        for j in range(self.boxsize):
          if self.Z[i][j] == 0:
            rect = patches.Rectangle(xy = (i - 0.5, j - 0.5), width = 1., height=1., color = 'gray')
            ax[k].add_patch(rect)
      # plt.axis('off')
    # plt.axis("equal")

    if show_colorbar:
      fig.subplots_adjust(right=0.85)
      cbar_ax = fig.add_axes([0.9, 0.15, 0.04, 0.7])
      fig.colorbar(im, cax=cbar_ax)
    return fig, ax

  def plot_2surf(self, fig=None, ax=None, show_colorbar=True):
    if fig == None:
      fig, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=150)
    indices = ['x', 'y', 'z']
    # ax.axis('off')
    [axi.set_axis_off() for axi in ax.ravel()]
    # plt.axis('off')
    zmin, zmax = np.min(self.zz), np.max(self.zz)

    for k in range(2):
      im = ax[k].imshow(self.zz[k], interpolation='spline36',
                        resample=True, extent=self.extent,
                        vmin=zmin, vmax=zmax, aspect='auto', cmap='viridis')
      ax[k].set_title(indices[k])
      for i in range(self.boxsize):
        for j in range(self.boxsize):
          if self.Z[i][j] == 0:
            rect = patches.Rectangle(
                xy=(i - 0.5, j - 0.5), width=1., height=1., color='gray')
            ax[k].add_patch(rect)
      # plt.axis('off')
    # plt.axis("equal")

    if show_colorbar:
      fig.subplots_adjust(right=0.85)
      cbar_ax = fig.add_axes([0.9, 0.15, 0.04, 0.7])
      fig.colorbar(im, cax=cbar_ax)
    return fig, ax

  def get_true_z(self, poses):
    # get map data from input points
    # rotation not counted

    x = poses[:, :2]
    z = np.array([f(x[:, 0], x[:, 1]) for f in self.z_true[:2]])
    return z
 

  def true_obs(self, poses, Q):
    '''
    use 2 dim representation of magn field for simlpicity

    get [th, magn] observations from 
      observation function h[x, map]
    '''
    x = poses[:2]
    z = np.array([f(x[0], x[1]) for f in self.z_true[:2]])
    if Q is not None:
      noise = np.random.multivariate_normal(
            np.zeros(2), Q, size=z.shape[1]).T
      z += noise
    # bx, by, bz

    theta = np.arctan2(z[1, :], z[0, :]) - poses[2]
    # from global to local coordinate frame
    magn = np.linalg.norm(z, axis=0)
    return np.vstack((theta, magn))

  def obs_by_rot(self, x):
    # convert from global to local frame
    # observation model
    z = np.array([f(x[0], x[1]) for f in self.z_true[:3]])
    def rotation(vector, theta):
      """Rotates 2-D vector"""
      R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
      return np.dot(R, vector)
    # rotate by -theta
    z[:2] = np.array([rotation(z[:2, i], -x[2, i]) for i in range(x.shape[1])]).T
    return z

  @staticmethod
  def obs_2global(z, th):
    # rot z by theta in xy plane
    # z = single obs
    # th = (n,) pts from particle filter
    def rotation(vector, theta):
      """Rotates 3-D vector"""
      R = np.array([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.],
                    [0., 0., 1.]])
      return np.dot(R, vector)
    # rotate by -theta
    z_gl = np.array([rotation(z, thi) for thi in th]).T
    return z_gl

  @staticmethod
  def obs2vec(x, z):
    # from local to global frame
    theta = z[0] + x[:, 2]
    magn = z[1]
    
    b = magn * np.array([np.cos(theta), np.sin(theta)])
    return b

    # self.trajs = pd.DataFrame(noisy_dat)

  # def sense_landmarks(state, field_map, max_observations):

  #   pass

  # def sense_magnetic(state, map):
  #   pass

  # def sense_visual(state):
  #   pass

  def update_weights(self, x, z):
    '''
    update weights procedure for pf
    compare coordinate observations to known observations
    '''
    ztrue = np.array([f(x[:, 0], x[:, 1]) for f in self.z_true[:3]])
    z_rec = Zspace.obs_2global(z[np.newaxis].T, x[:, 2])[0]
    dz = np.linalg.norm(ztrue - z_rec, axis=0)
    w = gaussian.pdf(dz, loc=0, scale= 1)

    # w = np.ones(x.shape[0])
    c = np.array(x[:, :2] + [0.5, 0.5]).astype(int).T
    sel = self.Z[c[0], c[1]] == 0
    w[sel] = 0.
    return w

  def update_weights_uniform(self, x, z):
    '''
    update weights procedure for pf
    compare coordinate observations to known observations
    '''
    # ztrue = np.array([f(x[:, 0], x[:, 1]) for f in self.z_true[:3]])
    # z_rec = Zspace.obs_2global(z[np.newaxis].T, x[:, 2])[0]
    # dz = np.linalg.norm(ztrue - z_rec, axis=0)
    # w = gaussian.pdf(dz, loc=0, scale=2)

    w = np.ones(x.shape[0])
    c = np.array(x[:, :2] + [0.5, 0.5]).astype(int).T
    sel = self.Z[c[0], c[1]] == 0
    w[sel] = 0.
    return w

def wrap_angle(angle):
    """
    Wraps the given angle to the range [-pi, +pi].

    :param angle: The angle (in rad) to wrap (can be unbounded).
    :return: The wrapped angle (guaranteed to in [-pi, +pi]).
    """

    pi2 = 2 * np.pi
    while angle < -np.pi:
        angle += pi2
    while angle >= np.pi:
        angle -= pi2

    return angle

wrap_angle_vec = np.vectorize(wrap_angle)

#%%
if __name__ == "__main__":
  zs = Zspace(boxsize = 11)
  print(zs.Z.T)
  # zs.plot_surf()
  # plt.show(block = True)

  # fig, ax = plt.subplots(1, 3, figsize=(10, 3), dpi=150)
  # zs.plot_3surf(fig, ax, True)

  fig, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=150)
  zs.plot_2surf(fig, ax, True)
  plt.show(block=True)


  # test observation model vec >> polar >> obs >> vec

  pts = np.random.multivariate_normal(np.zeros(3), 1e-12 * np.eye(3, 3), 100)
  # observation Bx By noise variance
  obs_noise_tol = [0.05, 0.05]
  zQ_ = np.diag(obs_noise_tol)
  obs = zs.true_obs(pts, zQ_)
  # reconstructed
  z_rec = zs.obs2vec(pts, obs)

  z_true = zs.get_true_z(pts)

  # delta = np.abs((z_true - z_rec) / z_true)
  err = np.linalg.norm(z_true - z_rec) / pts.shape[0]

  # cumulative proportional error of reconstruction
  print('err', err)
  # print(np.sum(delta) / pts.shape[0])
  # print('delta',(z_true - z_rec))

# #%%
# zs = Zspace(boxsize=11)
# # zs = sim.zs
# print(zs.Z.T)
# # zs.plot_surf()
# # plt.show(block = True)

# fig, ax = plt.subplots(1, 3, figsize=(10, 3), dpi=150)
# zs.plot_3surf(fig, ax, True)
# plt.show(block=True)

# %%

# zs = Zspace(boxsize=11)
# pts = np.random.multivariate_normal(np.zeros(3), 1e-12 * np.eye(3, 3), 10)
# z = zs.true_obs(pts)

# %%
