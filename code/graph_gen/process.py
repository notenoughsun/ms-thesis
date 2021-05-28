#%%

# import json
from functools import partial
import contextlib

import os.path
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import animation as anim
from progress.bar import FillingCirclesBar

# import matplotlib.patches as patches
# from numpy.core.numeric import indices
# import scipy.interpolate as interpolate
# from scipy.interpolate import LinearNDInterpolator
# from scipy.interpolate import CloughTocher2DInterpolator
# from scipy.interpolate import interpolate
# from matplotlib.patches import Rectangle

from tools.objects import FilterTrajectory, Gaussian
from tools.plot import plot2dcov
from filters.pf import PF
# import filters.pf as PF

from zspace import Zspace, wrap_angle


@contextlib.contextmanager
def get_dummy_context_mgr():
    """
    :return: A dummy context manager for conditionally writing to a movie file.
    """
    yield None


class Object(object):
    pass


# boxsize = 11
# zs = Zspace(boxsize)
# fig, ax = zs.plot_3surf()

class Process():
  def __init__(self, zs=None, boxsize=9):
    if zs is None:
      self.zs = Zspace(boxsize)
    else:
      self.zs = zs
      boxsize = None

    self.mg = self.zs

    # processed data dataframe
    self.df = None

    # params
    self.params = Object()
    params = self.params

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    self.path = os.path.dirname(os.path.abspath(filename))

    params.movie_fps = 10
    params.write_movie = True
    params.show_plots = True
    # params.plot_pause_len = 0.01
    params.plot_pause_s = 0.01
    params.output_dir = os.path.join(self.path, 'out/')
    params.movie_file = os.path.join(self.path, 'out/out.mp4')
    
    params.store_sim_data = False
    params.echo_debug_observations = False

    params.animate = False
    params.n_routes = 2 * len(self.mg.nodes)
    params.len_routes = 4

    params.dt = 0.1

    params.alphas = np.array([0.05, 0.001, 0.05, 0.01])
    params.betas = [0.05, 0.001, 0.1]
    params.num_particles = 60

    update_weights = self.zs.update_weights
    # update_weights = partial(Zspace.update_weights, self.zs)

    self.localization_filter = PF(None, params.alphas,
                                  params.betas, params.num_particles, update_weights)

  def plot_scene(self):
    fig, ax = self.zs.plot_3surf()
    return fig, ax

  def plot_traj(self, ax, traj, color='r'):
    ax.plot(traj[0, :], traj[1, :], linewidth=1, color=color)

  def generate_motion(self, traj):
    '''
    Generates the human motion trajectory
      using path given by precalculated trajectory.

    :param traj: All poses of human during moving
    (format: np.array([x, y, theta]))

    # :param max_obs_per_time_step: The maximum number of observations to generate per time step of the sim.
    :param alphas: The noise parameters of the control actions (format: np.array([a1, a2, a3, a4])).
    :param beta: The noise parameter of observations (format: np.array([range (cm), bearing (deg)])).
    :param dt: The time difference (in seconds) between two consecutive time steps.
    # :param animate: If True, this function will animate the generated data in a plot.
    :param plot_pause_s: The time (in seconds) to pause the plot animation between two consecutive frames.
    :return: SimulationData object.
    '''
    params = self.params
    # Q = np.diag([params.betax ** 2, params.betax ** 2, params.betath ** 2])

    # # [Bx, By, Bz]
    # observation_dim = 3

    # # State format: [x, y, theta]
    # state_dim = 3

    freq = 1. / (params.dt + 1e-10)

    x = np.array(traj)
    dx = np.array(x[:, 1:] - x[:, :-1])

    theta = np.arctan2(dx[1, :], dx[0, :])
    dtheta = theta[1:] - theta[:-1]
    dtheta = np.array([wrap_angle(angle) for angle in dtheta])

    # drot1 = np.append(0, dtheta)
    drot2 = np.append(dtheta, 0) + np.random.normal(0,
                                                    params.betas[1], dtheta.shape[0] + 1)
    drot1 = np.zeros_like(drot2) + np.random.normal(0,
                                                    params.betas[1], dtheta.shape[0] + 1)
    dtran = np.sqrt(dx[0, :]**2 + dx[1, :]**2) + \
        np.random.normal(0, params.betas[0], dtheta.shape[0] + 1)

    # theta = np.append(theta, theta[-1])
    # v = dx * freq
    # dtheta = theta[1:] - theta[:-1]
    # dtheta = np.array([wrap_angle(angle) for angle in dtheta])

    # acc = np.array(v[:,1:] - v[:, :-1]) * freq
    # ddtheta = dtheta[1:] - dtheta[:-1]

    # obs = np.vstack((acc, ddtheta))
    # poses = np.vstack((x, theta))

    # noise =  np.random.multivariate_normal(np.zeros(3), self.Q_, size = obs.shape[1]).T
    # nobs = obs + noise

    # true observations
    z = np.array([f(x[0], x[1]) for f in self.zs.z_true])

    # true motion
    u = drot1, dtran, drot2

    mean_prior = np.array([x[0, 0], x[1, 0], theta[0]]).T
    Sigma_prior = 1e-10 * np.eye(3, 3)

    data = u, z
    # (3, 99), (3, 100)

    initial_state = Gaussian(mean_prior, Sigma_prior)
    # return u, poses, z
    return data, initial_state

  def generate_input_data(self):
    '''
    prepare dataframe of true data, noisy data, initial gaussian state
    use for simulation
    '''
    self.routes = self.mg.generate_routes(
        self.params.n_routes, self.params.len_routes)

    dataset = []
    for traj in self.routes:
      # for oi in traj:
      data, initial_state = self.generate_motion(traj)
      dataset.append((traj, data, initial_state))

    df = pd.DataFrame(dataset, columns=['traj', 'data', 'x0'])
    self.df = df

  @staticmethod
  def plot_traj(traj, alpha=0.8, label = '', ax = None):
    if ax is not None:
      ax.plot(traj[0, :], traj[1, :], linewidth=1,
             alpha=alpha, label=label)
    else:
      plt.plot(traj[0, :], traj[1, :], linewidth=1, alpha=alpha, label = label, color = 'r')

  def plot_cond(self, traj, label = ''):
      # condition x on maze occupancy grid data
      c = np.array(traj + [[0.5], [0.5]]).astype(int)
      sel = self.zs.Z[c[0], c[1]] > 0

      Process.plot_traj(traj, label = label)
      # pts_sc = traj[:, sel]
      # plt.scatter(pts_sc[0, :], pts_sc[1, :], color='b')
      pts_sc = traj[:, np.logical_not(sel)]
      plt.scatter(pts_sc[0, :], pts_sc[1, :], color='r')
      # return sel

  def plot_data(self, fig=None, alpha=0.8):
    if fig == None:
      fig = plt.figure(figsize=(3, 3), dpi=150)
    self.mg.plot_field(fig)

    data = self.df[['traj']].to_numpy()[:, 0]

    for traj in data:
      Process.plot_traj(traj)
      # for oi in traj:
      # plt.plot(traj[0][0, :], traj[0][1, :], linewidth=1, alpha=alpha)
    plt.tight_layout()
    plt.axis('off')
    saveto = os.path.join(self.path, 'content/map_gen.png')
    plt.savefig(saveto)
    # plt.show(block = True)

    # self.zs.plot_surf()
    # self.zs.plot_walls(ax)

    # for mu in sim_trajectory.mean:
    # samples = mu.T
    # plt.plot(traj[0, :], traj[1, :], linewidth=3, color='r', label='true')

    # plt.cla()

    # if params.write_movie:
    #   plt.cla()
    #   # plot_field(z[1])
    #   #     plot_robot(data.debug.real_robot_path[t])
    #   #     plot_observation(data.debug.real_robot_path[t],
    #   #                      data.debug.noise_free_observations[t],
    #   #                      data.filter.observations[t])

    # if params.show_plots:
    #     plt.show(block=True)

    # if params.show_pose_error_plots:
    #   pass
    #     # fig, ax = plt.subplots(3, 1)
    #     # err = data.debug.real_robot_path - sim_trajectory.mean
    #     # t = np.array(range(data.num_steps))
    #     # sigma = sim_trajectory.covariance

    #     # ciplot(ax[0], t, err[:, 0], None, 3 * np.sqrt(sigma[0, 0, :]), None, label = r'$\| \Delta x \|$', color=None)
    #     # ciplot(ax[1], t, err[:, 1], None, 3 * np.sqrt(sigma[1, 1, :]), None, label = r'$\| \Delta y \|$', color=None)
    #     # ciplot(ax[2], t, wrap_angle_vec(err[:, 2]), None, 3 * np.sqrt(sigma[2, 2, :]), None, label = r'$\| \Delta \theta \|$', color=None)

    #     # fig.suptitle("pose error vs time")
    #     # plt.xlabel('time steps')
    #     # ax[0].set_ylabel(r'$\| \Delta x \|$')
    #     # ax[1].set_ylabel(r'$\| \Delta y \|$')
    #     # ax[2].set_ylabel(r'$\| \Delta \theta \|$')

    #     # plt.savefig(os.path.join(params.output_dir, params.movie_file + '.png'))
    #     # plt.close()
    #     # # plt.show()


  def run(self, input_data, fig = None):
    '''
    run the process cycle of filtering and localization
    '''
    params = self.params
    params.plot_pause_len = 0.01

    sim_data = []

    if fig is None:
      fig = plt.figure(figsize=(3, 3), dpi=150)
    # if True or params.show_plots or params.write_movie:
    #     fig = plt.figure(figsize=(3, 3), dpi=150)
    # if params.show_plots:
    #     plt.ion()

    if params.store_sim_data:
        if not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir)
        # save_input_data(data, os.path.join(params.output_dir, 'input_data.npy'))

    movie_writer = None
    if params.write_movie:
        get_ff_mpeg_writer = anim.writers['ffmpeg']
        metadata = dict(title='Localization Filter',
                        artist='matplotlib', comment='PS2')
        movie_fps = min(params.movie_fps, float(1. / params.plot_pause_len))
        movie_writer = get_ff_mpeg_writer(fps=movie_fps, metadata=metadata)
        movie_path = os.path.join(
            params.output_dir, params.movie_file + '.mp4')

    # progress_bar = FillingCirclesBar('Simulation Progress', max=data.num_steps)
    out_data = []
    for index, row in input_data.iterrows():
        oi, data, initial_state = row
        # ax = plt.cla()
        # Process.plot_traj(oi, label = 'true')

        sim_trajectory = self.localization_filter.filter(
            data, initial_state, alphas=params.alphas, betas=params.betas)
        # self.localization_filter.plot(sim_trajectory)
        # Process.plot_traj(oi)
        traj = sim_trajectory.mean[:, :2, 0].T
        Process.plot_traj(traj, label='sim')

        sim_data.append(sim_trajectory)

        # PF.plot(sim_trajectory)

    return sim_data

    # self.plot_data(fig)
    # # with movie_writer.saving(fig, movie_path, dpi=200) if params.write_movie else get_dummy_context_mgr():
    #     for index, row in self.df.iterrows():
    #       oi, data, initial_state = row

    #       ax = plt.cla()
    #       Process.plot_traj(oi)
  
    #       sim_trajectory = self.localization_filter.filter(data, initial_state)
    #       # self.localization_filter.plot(sim_trajectory)
    #       Process.plot_traj(oi)
    #       PF.plot(sim_trajectory)
    #     self.plot_data(fig)

        
    #     if params.echo_debug_observations:
    #         print("echo_debug_observations",
    #               data.debug.noise_free_observations[0, :])

    #     if params.store_sim_data:
    #         file_path = os.path.join(params.output_dir, 'output_data.npy')
    #         with open(file_path, 'wb') as data_file:
    #             np.savez(data_file,
    #                      mean_trajectory=sim_trajectory.mean,
    #                      covariance_trajectory=sim_trajectory.covariance)

  # def save_data(self, fname= 'content/noisy_dat.json'):
  #   saveto = os.path.join(self.path, fname)

  #   self.trajs.to_json(saveto, orient='values')
  #   # print("saveto:", saveto)

  # def load_data(self, fname='content/noisy_dat.json'):
  #   saveto = os.path.join(self.path, fname)
  #   self.trajs = pd.read_json(saveto, orient='values')

    # data_read = df['noisy_dat'].to_numpy()
    # with open(saveto, 'w') as w_file:
    #   # file.write(data)
    #   json.dump(noisy_dat, w_file)

    # with open(saveto, "r") as read_file:
    #   data_read = json.load(read_file)

# to use
# https://shapely.readthedocs.io/en/latest/manual.html#collections


# def save_data(data, file_path):
#     """
#     Saves the simulation's input data to the given filename.

#     :param data: A tuple with the filter and debug data to save.
#     :param file_path: The the full file path to which to save the data.
#     """

#     output_dir = os.path.dirname(file_path)
#     print(output_dir, file_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     with open(file_path, 'wb') as data_file:
#         np.savez(data_file,
#                  num_steps=data.num_steps,
#                  noise_free_motion=data.filter.motion_commands,
#                  real_observations=data.filter.observations,
#                  noise_free_observations=data.debug.noise_free_observations,
#                  real_robot_path=data.debug.real_robot_path,
#                  noise_free_robot_path=data.debug.noise_free_robot_path)

#%%

if __name__ == "__main__":
    boxsize = 11
    zs = Zspace(boxsize)

    sim = Process(zs = zs)
    sim.params.n_routes = 26

    sim.generate_input_data()
    # sim.plot_data(alpha=0.6)

    data = sim.df[:1]
    traj = sim.df[['traj']].to_numpy()[0, 0]

    print("plot field and single traj")
    fig = plt.figure(figsize=(3, 3), dpi=150)
    sim.mg.plot_field(fig)
    sim.plot_traj(traj, label = 'true')
    plt.legend()

    sim_data = sim.run(data, fig=fig)
    for row in sim_data:
      traj = row.mean[:, :2, 0].T
      sim.plot_cond(traj)
      print(row.mean.shape)
      for pts in row.mean:
        plt.scatter(pts[0, :], pts[1, :], s  = 0.3)
    # filter, no resampling

    # sim.mg.generate_routes(26)

    # zs = sim.zs
    # print(zs.Z.T)
    # zs.plot_surf()
    # plt.show(block = True)

    zs.plot_2surf()

# sim = Process(zs=zs)
# sim.params.n_routes = 30

# sim.generate_input_data()
# # sim.plot_data(alpha=0.6)

#%%
# boxsize = 11
# zs = Zspace(boxsize)

# sim = Process(zs=zs)
# sim.params.n_routes = 26

# sim.generate_input_data()


# sim = Process(zs = zs)
# sim.params.n_routes = 26

# sim.generate_input_data()

# fig = plt.figure(num="field3", figsize=(3, 3), dpi=150)

# data = sim.df[:1]

# # sim.plot_data(alpha=0.4)
# # sim.plot_scene(fig)
# # def plot_some(mg):
# fig = plt.figure(figsize=(3, 3), dpi=150)
# plt.axis('off')
# sim.zs.plot_field(fig)
# sim.run(data, fig = fig)

  # nx.draw(mg.G, pos=mg.pos, alpha=0.2, node_size=5, width=2)

# plot_some(sim.zs)
# plt.legend()

# plt.tight_layout()
# plt.axis('off')
# saveto = os.path.join(sim.path, 'content/sim_test.png')
# plt.savefig(saveto)
# plt.show()

# fig, ax = sim.plot_scene()

# %%
