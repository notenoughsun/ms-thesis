"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle

# tag::ekf[]
class EKF(LocalizationFilter):
    def predict(self, u):
        def state_jacobian(u, theta):
            drot1, dtran, drot2 = u
            G = np.eye(3)
            G[: 2, 2] = [
                -dtran * np.sin(theta + drot1), 
                dtran * np.cos(theta + drot1)]
            return G
        
        def control_jacobian(u, theta):
            drot1, dtran, drot2 = u
            phase = theta + drot1
            V = np.array([
                [-dtran * np.sin(phase), np.cos(phase), 0.],
                [dtran * np.cos(phase),  np.sin(phase), 0.],
                [1., 0., 1.]
            ])
            return V

        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        mu = get_prediction(self.mu, u)
        
        # calculate jacobians
        G = state_jacobian(u, mu[2])
        V = control_jacobian(u, mu[2])

        M = get_motion_noise_covariance(u, self._alphas)

        self._state_bar.mu = mu
        self._state_bar.Sigma = G @ self._state.Sigma @ G.T + V @ M @ V.T

    def update(self, z):
        # implement correction step         
        def measurement_jacobian(lm_id):
            mu = self._state_bar.mu
            dx = self._field_map.landmarks_poses_x[lm_id] - mu[0]
            dy = self._field_map.landmarks_poses_y[lm_id] - mu[1]
            q = dx**2 + dy**2
            H = np.array([[dy / q, -dx/q, -1.]])
            return H

        lm_id = np.int(z[1])
        H = measurement_jacobian(lm_id)

        cov = self.Sigma_bar
        S = H @ cov @ H.T + self._Q
        K = cov @ H.T @ np.linalg.inv(S)

        mu = self._state_bar.mu

        bearing = get_expected_observation(self._state_bar.mu, lm_id)[0]
        self._state.mu = mu[np.newaxis].T +  K * wrap_angle(z[0] - bearing)
        self._state.Sigma = (np.eye(3) - K @ H) @ cov

# end::ekf[]

