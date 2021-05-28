"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

# from code.graph_gen.tools.task import get_prediction
import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

import matplotlib.pyplot as plt
from matplotlib import animation as anim

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
# from tools.task import sample_from_odometry
from tools.task import wrap_angle


from tools.objects import FilterTrajectory, Gaussian

# tag::pf[]
class PF(LocalizationFilter):
    def __init__(self, initial_state, \
        alphas, betas, num_particles, update_w):

        if initial_state is None:
            initial_state = Gaussian(np.zeros((3, 1)), 1e-12 * np.eye(3, 3))

        super(PF, self).__init__(initial_state, alphas, betas)
        
        self.num_particles = num_particles
        
        self.x = np.random.multivariate_normal(initial_state.mu.squeeze(), initial_state.Sigma, num_particles)
        self.w = np.ones(num_particles) / num_particles
        
        self._state = initial_state
        self._state_bar = initial_state
        # map & accept/reject function
        self.update_w = update_w
        # self.field_map = field_map

    def reinit(self, initial_state, alphas, betas):
        super(PF, self).__init__(initial_state, alphas, betas)
        self.x = np.random.multivariate_normal(initial_state.mu.squeeze(), initial_state.Sigma, self.num_particles)
        self.w = np.ones(self.num_particles) / self.num_particles
        
        self._state = initial_state
        self._state_bar = initial_state


    def get_prediction(state, motion):
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

        assert state.shape == (3,)
        assert motion.shape == (3,)

        x, y, theta = state
        drot1, dtran, drot2 = motion

        theta = wrap_angle(theta + drot1)
        # theta = wrap_angle(theta)
        x += dtran * np.cos(theta)
        y += dtran * np.sin(theta)
        # Wrap the angle between [-pi, +pi].
        theta = wrap_angle(theta + drot2)

        return np.array([x, y, theta])

    def predict(self, u):
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma

        for i in range(self.num_particles):
            # sample_from_odometry using transition model
            self.x[i] = PF.get_prediction(self.x[i], u)
        
        self._state_bar = get_gaussian_statistics(self.x)


    def update(self, z):
        # TODO implement correction step

        # def update_w(x): 
        #     dz = wrap_angle(z[0] - get_observation(x, z[1])[0] )
        #     w = gaussian.pdf(dz, loc=0, scale=np.sqrt(self._Q))
        #     return w
        self.w = self.update_w(self.x[:, :2], z)
        # self.w = np.array(list(map(self.update_w, self.x, z)))
        self.w /= np.sum(self.w)

        def low_variance_sampling(w, n):
            x_new = []
            r = np.random.uniform(0., 1. / n)
            c = w[0]
            i = 0
            for m in range(n):
                u = r + m/n
                while u > c:
                    i = (i + 1) % n
                    c+= w[i]
                x_new.append(self.x[i])
            return np.array(x_new)

        self.x = low_variance_sampling(self.w, self.num_particles)        
        
        self.w = [1. / self.num_particles] * self.num_particles

        self._state = get_gaussian_statistics(self.x)

        self.sim_trajectory = None

    def filter(self, data, initial_state, alphas, betas):
        self.reinit(initial_state, alphas, betas)
        u, z = data
        u = np.array(u).T
        z = np.array(z).T
        num_steps = u.shape[0]
        
        mean_trajectory = np.zeros((num_steps, self.state_dim, self.num_particles))
        sim_trajectory = FilterTrajectory(mean_trajectory)
        sim_trajectory.covariance = np.zeros((self.state_dim,
                                                self.state_dim,
                                                num_steps))

        for t in range(num_steps):
            # tp1 = t + 1
            # print("time", tp1)

            # # Control at the current step.
            # # Observation at the current step.
            # u, z = data[t]

            self.predict(u[t])
            self.update(z[t])

            # self._state_bar.mu = self.mu[np.newaxis].T
            # self._state_bar.Sigma = self.Sigma

            sim_trajectory.mean[t, :, :] = self.mu[np.newaxis].T
            sim_trajectory.covariance[:, :, t] = self.Sigma

        # self.sim_trajectory = sim_trajectory
        return sim_trajectory

    @staticmethod
    def plot(sim_trajectory, show_particles = True):
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

class Object(object):
    pass


if __name__ == "__main__":
    params = Object()
    params.alphas = np.array([0.05, 0.001, 0.05, 0.01])
    params.betas = [0.05, 0.001, 0.1]
    # params.betath = 0.1
    params.num_particles = 100
    update_weights = None
    localization_filter = PF(None, params.alphas,
                             params.betas, params.num_particles, update_weights)
