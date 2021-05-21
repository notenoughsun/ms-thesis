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

class Object(object):
    pass


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

  def obs(self, state):
    '''
    state = [x, y, theta]
    return observations in global frame
    '''
    z = [self.z_true[i](state[0], state[1]) for i in range(3)]
    return z

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

  def plot_3surf(self, show_colorbar = True):
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


  def update_weights(self, x, z):
    # def update_w(x): 
    #     dz = wrap_angle(z[0] - get_observation(x, z[1])[0] )
    #     w = gaussian.pdf(dz, loc=0, scale=np.sqrt(self._Q))
    #     return w
    w = np.ones(x.shape[0])
    return w
    # plt.show()
    # # self.draw_graph()

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

class Simulation():
  def __init__(self, boxsize = 9):

    self.zs = Zspace(boxsize)
    self.mg = self.zs
    # self.mg = Mgraph(self.zs)

    # params
    self.params = Object()
    params = self.params

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    self.path = os.path.dirname(os.path.abspath(filename))

    params.movie_fps = 10
    params.write_movie = True
    params.show_plots = True
    # params.plot_pause_len
    params.output_dir = os.path.join(self.path, 'out/')
    params.movie_file = os.path.join(self.path, 'out/out.mp4')
    params.store_sim_data = True
    
    params.animate = False
    params.plot_pause_s = 0.01
    params.n_routes = 2 * len(self.mg.nodes)
    params.len_routes = 4

    params.dt = 0.1

    params.alphas = np.array([0.05, 0.001, 0.05, 0.01])
    params.betas = [0.05, 0.001, 0.1]
    # params.betath = 0.1
    params.num_particles = 100

    mean_prior = np.array([0., 0., 0.])
    Sigma_prior = 1e-12 * np.eye(3, 3)
    initial_state = Gaussian(mean_prior, Sigma_prior)

    self.localization_filter = PF(initial_state, params.alphas, \
      params.betas, params.num_particles, Zspace.update_weights, self.zs)


  def plot_scene(self):
    fig, ax = self.zs.plot_3surf()
    return fig, ax
    # plt.show()
  
  def plot_traj(self, ax, traj, color = 'r'):
    ax.plot(traj[0, :], traj[1, :], linewidth = 1, color = color)

  # def generate_motion(t, dt):
  #   pass

  # def sense_landmarks(state, field_map, max_observations):
    
  #   pass

  # def sense_magnetic(state, map):
  #   pass

  # def sense_visual(state):
  #   pass

  def generate_routes(self, n_routes):
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
    
    mg = self.mg
    # G = mg.G
    # create routes enough for coverage
    if n_routes == None:
      n_routes = 2 * len(mg.nodes)

    selected = np.random.randint(0, len(mg.nodes), size= n_routes)
    routes =  np.array_split(selected, n_routes // params.len_routes)  
    datasetgt,_ = mg.gen_routes(routes)

    noisy_dat = []

    for tr in datasetgt:
      nt = NoisyTraj(tr)
      if nt.curve == None:
        continue

      obs = nt.data_gen(repeat=1)
      noisy_dat.append(obs)

    self.data = noisy_dat
    # self.trajs = pd.DataFrame(noisy_dat)

  def generate_input_data(self, traj):
    params = self.params
    # Q = np.diag([params.betax ** 2, params.betax ** 2, params.betath ** 2])

    # # [Bx, By, Bz]
    # observation_dim = 3

    # # State format: [x, y, theta]
    # state_dim = 3

    freq = 1./ (params.dt + 1e-10)

    x = np.array(traj)
    dx = np.array(x[:, 1:] - x[:, :-1])

    theta = np.arctan2(dx[1, :], dx[0, :])
    dtheta = theta[1:] - theta[:-1]
    dtheta = np.array([wrap_angle(angle) for angle in dtheta])

    # drot1 = np.append(0, dtheta)
    drot2 = np.append(dtheta, 0) + np.random.normal(0, params.betas[1], dtheta.shape[0] + 1)
    drot1 = np.zeros_like(drot2) + np.random.normal(0, params.betas[1], dtheta.shape[0] + 1)
    dtran = np.sqrt(dx[0, :]**2 + dx[1, :]**2) + np.random.normal(0, params.betas[0], dtheta.shape[0] + 1)

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

    mean_prior = np.array([x[0, 0], x[1, 0], theta[0]])
    Sigma_prior = 1e-10 * np.eye(3, 3)

    data = u, z
    initial_state = Gaussian(mean_prior, Sigma_prior)
    # return u, poses, z
    return data, initial_state

  def plot_data(self, fig = None, alpha = 0.4):
    if fig == None:
      fig = plt.figure(figsize=(3, 3), dpi=150)
    self.mg.plot_field(fig)

    for traj in self.data:
      for oi in traj:
        plt.plot(oi[0, :], oi[1, :], linewidth = 1, alpha = alpha)

    plt.tight_layout()
    plt.axis('off')
    saveto = os.path.join(self.path, 'content/map_gen.png')
    plt.savefig(saveto)
    # plt.show(block = True)

  def run(self):
    '''
    run the process cycle of filtering and localization
    '''
    params = self.params
    params.plot_pause_len = 0.01

    fig = None
    if params.show_plots or params.write_movie:
        fig = plt.figure(figsize=(5, 5), dpi=150)
    if params.show_plots:
        plt.ion()

    # load or generate data
    traj = self.data[0][0]
    
    # end
    
    if params.store_sim_data:
        if not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir)
        # save_input_data(data, os.path.join(params.output_dir, 'input_data.npy'))

    movie_writer = None
    if params.write_movie:
        get_ff_mpeg_writer = anim.writers['ffmpeg']
        metadata = dict(title='Localization Filter', artist='matplotlib', comment='PS2')
        movie_fps = min(params.movie_fps, float(1. / params.plot_pause_len))
        movie_writer = get_ff_mpeg_writer(fps=movie_fps, metadata=metadata)
        movie_path = os.path.join(params.output_dir, params.movie_file + '.mp4')

    # progress_bar = FillingCirclesBar('Simulation Progress', max=data.num_steps)
    
    with movie_writer.saving(fig, movie_path, dpi = 200) if params.write_movie else get_dummy_context_mgr():
        data, initial_state = self.generate_input_data(traj)
        sim_trajectory = self.localization_filter.filter(data, initial_state)
        # self.localization_filter.plot(sim_trajectory)

        ax = plt.cla()

        self.plot_data(fig)

        # self.zs.plot_surf()
        # self.zs.plot_walls(ax)

        # for mu in sim_trajectory.mean:
          # samples = mu.T
        plt.plot(traj[0, :], traj[1, :], linewidth = 3, color = 'r', label = 'true')

        for i in range(0, params.num_particles, 6):
          plt.plot(sim_trajectory.mean[:, 0, i], sim_trajectory.mean[:, 1, i], linewidth = 1)

        for t in range(0, sim_trajectory.mean.shape[0], 2):
          plot2dcov(sim_trajectory.mean[t, :-1, 0],
                          sim_trajectory.covariance [:-1, :-1, t],
                          'red', 0.5)

        plt.legend()
        plt.show(block = True)
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
        
        echo_debug_observations = False
        if echo_debug_observations:
            print("echo_debug_observations", data.debug.noise_free_observations[0, :])

        if params.store_sim_data:
            file_path = os.path.join(params.output_dir, 'output_data.npy')
            with open(file_path, 'wb') as data_file:
                np.savez(data_file,
                        mean_trajectory=sim_trajectory.mean,
                        covariance_trajectory=sim_trajectory.covariance)





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



if __name__ == "__main__":
  sim = Simulation(11)
  sim.generate_routes(26)
  # sim.save_data()
  # sim.plot_data()
  # plt.show()
  # plt.close()

  # zs = Zspace(boxsize = 11)
  zs = sim.zs
  # print(zs.Z.T)
  # # zs.plot_surf()
  # # plt.show(block = True)
  # zs.plot_3surf()

  routes = sim.data

  # traj = routes[0][0]
  # nobs, poses, obs = sim.generate_input_data(traj)
  
  # fig, ax = sim.plot_scene()
  # for axi in ax:
  #   sim.plot_traj(axi, traj, color = 'r')
  # plt.show()

  sim.run()


  # def plot_scene(self):
  #   fig, ax = self.zs.plot_3surf()
  #   return fig, ax
  #   # plt.show()
  
  # def plot_traj(self, ax, traj):
  #   ax.plot(traj[0, :], traj[1, :], linewidth = 1)