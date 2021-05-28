#%%
from process import Process, Object
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import animation as anim
from progress.bar import FillingCirclesBar

from process import Process
from zspace import Zspace
from tools.task import wrap_angle
from tools.task import sample_from_odometry
from tools.task import get_gaussian_statistics
from scipy.stats import norm as gaussian
from numpy.random import uniform

from tools.plot import plot2dcov
# from tools.objects import FilterTrajectory

from filters.pf import PF
# from tools.objects import Gaussian


class Gaussian(object):
    """
    Represents a multi-variate Gaussian distribution representing the state of the robot.
    """

    def __init__(self, mu, Sigma, ndim = 6):
        """
        Sets the internal mean and covariance of the Gaussian distribution.

        :param mu: A 1-D numpy array (size 3x1) of the mean (format: [x, y, theta]).
        :param Sigma: A 2-D numpy ndarray (size 3x3) of the covariance matrix.
        """

        assert isinstance(mu, np.ndarray)
        assert isinstance(Sigma, np.ndarray)
        assert Sigma.shape == (ndim, ndim)

        if mu.ndim < 1:
            raise ValueError('The mean must be a 1D numpy ndarray of size 3.')
        elif mu.shape == (ndim,):
            # This transforms the 1D initial state mean into a 2D vector of size 3x1.
            mu = mu[np.newaxis].T
        elif mu.shape != (ndim, 1):
            raise ValueError('The mean must be a vector of size 3x1.')
        self.mu = mu
        self.Sigma = Sigma


class FilterTrajectory(object):
    def __init__(self, mean_trajectory, covariance_trajectory):
        assert isinstance(mean_trajectory, np.ndarray)
        assert isinstance(covariance_trajectory, np.ndarray)
        self.mean = mean_trajectory
        self.covariance = covariance_trajectory

class Filter(PF):
    '''
    wanna implement x,y,vx,vy,theta state & motion model
    '''
    def __init__(self, initial_state,
                 alphas=None, betas=None,
                 num_particles = 100, update_w = None):

        self._Q = np.diag(betas ** 2)
        initial_state = Gaussian(np.zeros((6, 1)), self._Q)
        assert isinstance(initial_state, Gaussian)

        self.state_dim = 6   # [x, y, theta, dx, dy, dtheta]

        # assert isinstance(initial_state, Gaussian)
        assert initial_state.Sigma.shape == (self.state_dim, self.state_dim)
        assert initial_state.mu.shape == (self.state_dim, 1)

        self.motion_dim = 3  # [drot1, dtran, drot2, d2rot1, d2tran, d2rot2]
        self.obs_dim = 3     # [bearing, bz]
        # self.obs_dim = 2     # [field power bx by bz]
        # self.obs_dim = 2     # [bearing, field power]

        # Filter noise parameters.
        self._alphas = alphas
        # Measurement variance.
        self.num_particles = num_particles

        self._state = initial_state
        self._state_bar = initial_state

        self.num_particles = num_particles

        self.x = np.random.multivariate_normal(
            initial_state.mu.squeeze(), initial_state.Sigma, num_particles)
        self.w = np.ones(num_particles) / num_particles

        self.update_w = update_w


    def reinit(self, initial_state, alphas, betas):
        # return self().__ini(initial_state, alphas, betas)

        assert isinstance(initial_state, Gaussian)
        assert initial_state.Sigma.shape == (self.state_dim, self.state_dim)

        if initial_state.mu.ndim < 1:
            raise ValueError(
                'The initial mean must be a 1D numpy ndarray of size 6.')
        elif initial_state.mu.shape == (self.state_dim, ):
            # This transforms the 1D initial state mean into a 2D vector of size 3x1.
            initial_state.mu = initial_state.mu[np.newaxis].T
        elif initial_state.mu.shape != (self.state_dim, 1):
            raise ValueError(
                'The initial state mean must be a vector of size 6x1')

        self.x = np.random.multivariate_normal(
            initial_state.mu.squeeze(), initial_state.Sigma, self.num_particles)

        self.w = np.ones(self.num_particles) / self.num_particles

        # Filter noise parameters.
        self._alphas = alphas
        # Measurement variance.
        self._Q = np.diag(betas ** 2)

        self._state = initial_state
        self._state_bar = initial_state
        self.dt = 0.1

    def get_prediction(self, state, motion):
        """
        Predicts the next state given state and the motion command.

        :param state: The current state of the robot (format: [x, y, theta]).
        :param motion: The motion command to execute (format: [drot1, dtran, drot2]).
        :return: The next state of the robot after executing the motion command
                (format: np.array([x, y, theta])). The angle will be in range
                [-pi, +pi].
        """
        # self.x[i] = sample_from_odometry(self.x[i], u, self._alphas)
        # self.x[i][2] = wrap_angle(self.x[i][2])

        assert isinstance(state, np.ndarray)
        assert isinstance(motion, np.ndarray)
        assert state.shape ==  (6,)
        assert motion.shape == (3,) #ax, ay, az

        # add Rigid Body Transformations
        x = np.zeros(6)
        x[:3] = state[:3] + self.dt * state[3:]
        x[3:] = state[3:] + self.dt * motion
        x[2] = wrap_angle(x[2])
        return x

    def predict(self, u):
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma

        self.x = np.array([self.get_prediction(xi, u) for xi in self.x])
        noise = np.random.multivariate_normal(
            np.zeros(self.x.shape[1]), self._Q, size=self.x.shape[0])
        # x_new = np.array(x_new)
        self.x += noise # e.g. sample from odometry

        self._state_bar = get_gaussian_statistics(self.x[:, :3])

    def update(self, z):
        fl = True
        # z_vec = Zspace.obs2vec(self.x[:2], z)

        # additionally rotate by theta
        # def rotation(vector, theta):
        #     """Rotates 2-D vector"""
        #     R = np.array([[np.cos(theta), -np.sin(theta)],
        #                     [np.sin(theta), np.cos(theta)]])
        #     return np.dot(R, vector)
        # z[:2] = rotation(z[np.newaxis].T, self.x[2]).T

        self.w = self.update_w(self.x, z)

        if np.sum(self.w) < 1e-10:
            fl = False
            print('bad localization, resample from gaussian')
            self.x[:, :3] = np.random.multivariate_normal(
                self.mu, self.Sigma, self.num_particles)
            self.w = np.ones(self.num_particles)

        self.w /= np.sum(self.w + 1e-10)

        def low_variance_sampling(w, n):
            x_new = []
            r = np.random.uniform(0., 1. / n)
            c = w[0]
            i = 0
            for m in range(n):
                u = r + m/n
                while u > c:
                    i = (i + 1) % n
                    c += w[i]
                x_new.append(self.x[i])
            return np.array(x_new)

        self.x = low_variance_sampling(self.w, self.num_particles)

        self.w = [1. / self.num_particles] * self.num_particles

        # self._state = get_gaussian_statistics(self.x)
        self._state = get_gaussian_statistics(self.x[:, :3])
        # self.sim_trajectory = None
        return fl

    def filter(self, data, initial_state, alphas, betas):
        self.reinit(initial_state, alphas, betas)
        u, z = data
        u = np.array(u).T
        z = np.array(z).T
        num_steps = u.shape[0]

        mu = np.zeros((num_steps, self.state_dim))
        # mu = np.zeros((num_steps, self.state_dim, self.num_particles))
        covariance = np.zeros((num_steps, self.state_dim, self.state_dim))
        sim_trajectory = FilterTrajectory(mu, covariance)

        sim_data_pred = []
        sim_data_upd = []

        for t in range(num_steps):
            # print('time = ', t)
            self.predict(u[t])
            sim_data_pred.append(self.x[:, :3])
            fl = self.update(z[t])

            sim_trajectory.mean[t, :3] = self.mu[np.newaxis]
            sim_trajectory.covariance[t, :3, :3] = self.Sigma

            if fl is False:
                return sim_trajectory, sim_data_pred, sim_data_upd, t
            sim_data_upd.append(self.x[:, :3])

        return sim_trajectory, sim_data_pred, sim_data_upd, num_steps

    @staticmethod
    def plot_state(x):
        plot2dcov(x.mean[:2], x.covariance[:2, :2])

    @staticmethod
    def plot(sim_trajectory, show_particles=True):
        if show_particles:
            for mu in sim_trajectory.mean:
                samples = mu.T
                plt.scatter(samples[0], samples[1], s=2)
        # plt.show()
# end::pf[]

# def plot_sim_trajectory():
#       for i in range(0, params.num_particles, 6):
#           plt.plot(sim_trajectory.mean[:, 0, i],
#                    sim_trajectory.mean[:, 1, i], linewidth=1)

#         for t in range(0, sim_trajectory.mean.shape[0], 2):
#           plot2dcov(sim_trajectory.mean[t, :-1, 0],
#                     sim_trajectory.covariance[:-1, :-1, t],
#                     'red', 0.5)

#         plt.legend()
#         plt.show(block=True)




class Sim(Process):
    def __init__(self, zs=None, boxsize=9, betas = None):
        super().__init__(zs=zs, boxsize=boxsize)
        
        self.params = Object()
        params = self.params

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

        # params.alphas = np.array([0.05, 0.001, 0.05, 0.01])
        # params.betas = [0.05, 0.001, 0.1]
        params.num_particles = 60

        params.dt = 0.1
        if betas is None:
            params.betas = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        else:
            params.betas = betas

        params.alphas = None

        self._Q = np.diag(params.betas)
        obs_noise_tol = [0.05, 0.05, 0.005]
        self._ZQ = np.diag(obs_noise_tol)
        self._MQ = np.diag(np.ones(3) * 1e-2)

        initial_state = Gaussian(np.zeros((6, 1)), self._Q)
    
        update_weights = self.zs.update_weights

        self.localization_filter = Filter(initial_state, params.alphas,
                                  params.betas, params.num_particles, update_weights)

    def generate_motion(self, traj):
        params = self.params
        # Q = np.diag([params.betax ** 2, params.betax ** 2, params.betath ** 2])

        freq = 1. / (params.dt + 1e-10)

        x = np.array(traj)
        dx = np.array(x[:, 1:] - x[:, :-1]) * freq
        acc = np.array(dx[:,1:] - dx[:, :-1]) * freq

        theta = np.arctan2(dx[1, :], dx[0, :])
        theta = np.append(theta, theta[-1])
        dtheta = theta[1:] - theta[:-1]
        dtheta = np.array([wrap_angle(angle) for angle in dtheta]) * freq

        ddtheta = (dtheta[1:] - dtheta[:-1]) * freq

        # acc += np.random.normal(0, params.betas[1], acc.shape[0])

        obs = np.vstack((acc, ddtheta))
        noise = np.random.multivariate_normal(
            np.zeros(obs.shape[0]), self._ZQ, size=obs.shape[1]).T
        obs += noise
        # noisy motion accelerometer noise

        poses = np.vstack((x, theta))

        # Q = np.diag(params.betas)
        # mean_prior = np.array([x[0, 0], x[1, 0], theta[0]]).T
        # Sigma_prior = 1e-10 * np.eye(3, 3)

        # initial_state = Object()
        # initial_state.mu = mu
        # initial_state.Sigma = Q

        mu = np.hstack((x[:, 0], theta[0], dx[:, 0], dtheta[0]))
        initial_state = Gaussian(mu, self._Q)
        # # true observations
        # z = np.zeros((2, x.shape[1]))

        # self.plot_scene()
        # fig = plt.figure(figsize=(3, 3), dpi=150)
        # self.mg.plot_field(fig)
        # plot2dcov(mu[:2], self._Q[:2, :2])
        # plt.show()
        
        z = self.zs.obs_by_rot(poses)
        z += np.random.multivariate_normal(np.zeros(3),
                                           self._MQ, size=z.shape[1]).T
        # magnetometer noise

        # z = self.zs.true_obs(poses, None)
        # z = self.zs.true_obs(poses, self._ZQ)
        data = obs, z

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

    def run(self, input_data, fig = None):
        '''
        run the process cycle of filtering and localization
        '''
        params = self.params
        params.plot_pause_len = 0.01

        if fig is None:
            fig = plt.figure(figsize=(3, 3), dpi=150)
        # if True or params.show_plots or params.write_movie:
        #     fig = plt.figure(figsize=(3, 3), dpi=150)
        # if params.show_plots:
        #     plt.ion()
        self.mg.plot_field(fig)

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
        sim_data = []
        for index, row in input_data.iterrows():
            oi, data, initial_state = row
            # ax = plt.cla()
            # Process.plot_traj(oi, label = 'true')

            # sim_trajectory = self.localization_filter.filter(
            #     data, initial_state, alphas=params.alphas, betas=params.betas)
            # self.localization_filter.plot(sim_trajectory)
            # Process.plot_traj(oi)

            sim_trajectory, sim_data_pred, sim_data_upd, num_steps = \
                sim.localization_filter.filter(
                data, initial_state, params.alphas, params.betas)
            # traj = sim_trajectory.mean[:, :2, 0].T
            # Process.plot_traj(traj, label='sim')

            sim_data.append(
                (sim_trajectory, sim_data_pred, sim_data_upd, num_steps))

            Process.plot_traj(oi, label='true')
            traj = sim_trajectory.mean[:num_steps, :2].T
            # sim.plot_cond(traj)
            Process.plot_traj(traj, label='sim')
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


# def sim1(sim, data, traj):
#     print("plot field and single traj")
#     fig = plt.figure(figsize=(5,5), dpi=150)
#     sim.mg.plot_field(fig)
#     sim.plot_traj(traj, label='true')
#     plt.legend()

#     sim_data = sim.run(data, fig=fig)
#     for sim in sim_data:
#         for i in range(0, sim.mean.shape[0]):
#             plt.scatter(sim.mean[i, 0, :], sim.mean[i, 1, :], s = 0.5)
#             mu = sim.mean[i, :2, 0]
#             cov = sim.covariance[:2, :2, i]
#             plot2dcov(mu, cov)

def test_filter(sim, input_data):
    alphas = sim.params.alphas
    betas = sim.params.betas

    for index, row in input_data.iterrows():
        oi, data, initial_state = row
        
        fig = plt.figure(figsize=(5, 5), dpi=150)
        # plot2dcov(initial_state.mu[:2, 0], initial_state.Sigma[:2, :2])
        sim_trajectory, sim_data_pred, sim_data_upd, num_steps = sim.localization_filter.filter(
            data, initial_state, alphas, betas)
        
        sim.mg.plot_field(fig)

        Process.plot_traj(oi, label = 'true')
        traj = sim_trajectory.mean[:num_steps, :2].T
        # sim.plot_cond(traj)
        Process.plot_traj(traj, label='sim')

        for x in sim_data_pred:
            plt.scatter(x[:, 0], x[:,1], s=0.5)
        plt.show()

        fig = plt.figure(figsize=(5, 5), dpi=150)
        sim.mg.plot_field(fig)
        Process.plot_traj(oi, label='true')
        traj = sim_trajectory.mean[:num_steps, :2].T
        # sim.plot_cond(traj)

        for x in sim_data_upd:
            plt.scatter(x[:, 0], x[:, 1], s=0.5)
        plt.show()

        fig, ax = sim.zs.plot_3surf()
        for axis in ax:
            for x in sim_data_upd:
                Process.plot_traj(oi, label='true', ax = axis)
                axis.scatter(x[:, 0], x[:, 1], s=0.5)
        plt.show()


def test_filter2(sim):
    traj = sim.mg.generate_routes(4, 4)
    traj = traj[0, :, 0:-1:2]
    dataset = []
    data, initial_state = sim.generate_motion(traj)
    dataset.append((traj, data, initial_state))
    df = pd.DataFrame(dataset, columns=['traj', 'data', 'x0'])
    # sim.localization_filter.params.n_parti

    for index, row in df.iterrows():
        oi, data, initial_state = row

        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111)
        plt.axis('off')

        Process.plot_traj(oi, label='true', ax=ax)
        # plot2dcov(initial_state.mu[:2, 0], initial_state.Sigma[:2, :2])
        sim_trajectory, sim_data_pred, sim_data_upd, num_steps = sim.localization_filter.filter(
            data, initial_state, alphas=None, betas=sim.params.betas)

        # sim.mg.plot_field(fig)

        traj = sim_trajectory.mean[:num_steps, :2].T
        Process.plot_traj(traj, label='sim', ax=ax)
        
        for x in sim_data_upd:
            plt.scatter(x[:, 0], x[:, 1], s=0.5)
        plt.savefig("new_odometry_pf.png")    
        plt.show()
        # sim.mg.plot_field(fig)

        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111)
        plt.axis('off')

        Process.plot_traj(oi, label='true', ax=ax)
        # plot2dcov(initial_state.mu[:2, 0], initial_state.Sigma[:2, :2])
        sim.localization_filter.update_w = sim.zs.update_weights_uniform

        sim_trajectory2, sim_data_pred2, sim_data_upd2, num_steps = sim.localization_filter.filter(
            data, initial_state, alphas=None, betas=sim.params.betas)
        traj2 = sim_trajectory2.mean[:num_steps, :2].T
        Process.plot_traj(traj, label='sim', ax=ax)
        Process.plot_traj(traj2, label='sim_uniform', ax=ax)
        # sim.mg.plot_field(fig)
        plt.legend()
        for x in sim_data_upd2:
            plt.scatter(x[:, 0], x[:, 1], s=0.5)
        plt.savefig("new_odometry_pf_uniform.png")

        plt.show()



if __name__ == "__main__":
    boxsize = 11
    zs = Zspace(boxsize)

    betas = np.array([5e-2, 5e-2, 1e-4, 1e-4, 1e-4, 1e-4])
    sim = Sim(zs=zs, betas = betas)
    sim.params.n_routes = 5

    sim.generate_input_data()

    data = sim.df[:3]
    # traj = sim.df[['traj']].to_numpy()[0, 0]

    # sim1(sim, data, traj)
    # test_filter(sim, sim.df[:1])
    test_filter2(sim)

    fig = plt.figure(figsize=(5, 5), dpi=150)
    sim.run(data, fig)
    plt.legend()
    plt.show()

# #%%
# from zspace import Zspace

# zs = Zspace(boxsize= 11)
# pts = np.random.multivariate_normal(np.zeros(3), 1e-12 * np.eye(3, 3), 10)
# z = zs.true_obs(pts)


#%%
