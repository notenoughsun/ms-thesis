from scipy.interpolate import interpolate
# from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from routes import Mgraph

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt


def generate_motion(t, dt):
  pass


def sense_landmarks(state, field_map, max_observations):
  
  pass

def sense_magnetic(state, map):
  pass

def sense_visual(state):
  pass



def generate_data(traj,
                  # num_landmarks_per_side,
                  # max_obs_per_time_step,
                  alphas,
                  beta,
                  dt,
                  # animate=False,
                  plot_pause_s=0.01):
  pass
  """
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
    """
