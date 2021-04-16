from scipy.interpolate import interpolate
# from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt

from routes import Mgraph
from NoisyTraj import NoisyTraj

import json
import os.path
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Simulation():
  def __init__(self, boxsize = 9):
    # boxsize = 9
    # G = None
    # Z = None
    # super().__init__(boxsize, G, Z)

    self.mg = Mgraph(boxsize)
    # self.G = self.mg.G

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    self.path = os.path.dirname(os.path.abspath(filename))

  # def generate_motion(t, dt):
  #   pass


  # def sense_landmarks(state, field_map, max_observations):
    
  #   pass

  # def sense_magnetic(state, map):
  #   pass

  # def sense_visual(state):
  #   pass

  def generate_data(self):
    # def generate_data(traj,
    #                   # num_landmarks_per_side,
    #                   # max_obs_per_time_step,
    #                   alphas,
    #                   beta,
    #                   dt,
    #                   # animate=False,
    #                   plot_pause_s=0.01):
    # pass
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
    
    mg = self.mg
    # G = mg.G
    # create routes enough for coverage
    n = 2 * len(mg.nodes)
    selected = np.random.randint(0, len(mg.nodes), size= n)
    routes =  np.array_split(selected, n // 4)  
    datasetgt,_ = mg.gen_routes(routes)

    noisy_dat = []

    for tr in datasetgt:
      nt = NoisyTraj(tr)
      if nt.curve == None:
        continue

      obs = nt.data_gen(repeat=1)
      noisy_dat.append(obs)

    self.data = noisy_dat
    self.trajs = pd.DataFrame(noisy_dat)


  def save_data(self, fname= 'content/noisy_dat.json'):
    saveto = os.path.join(self.path, fname)
    
    self.trajs.to_json(saveto, orient='values')
    # print("saveto:", saveto)

  def load_data(self, fname='content/noisy_dat.json'):
    saveto = os.path.join(self.path, fname)  
    self.trajs = pd.read_json(saveto, orient='values')
    
    # data_read = df['noisy_dat'].to_numpy()
    # with open(saveto, 'w') as w_file:
    #   # file.write(data)
    #   json.dump(noisy_dat, w_file)

    # with open(saveto, "r") as read_file:
    #   data_read = json.load(read_file)
    

  def plot_data(self, fig = None):
    if fig == None:
      fig = plt.figure(figsize=(3, 3), dpi=150)
    self.mg.plot_field(fig)

    for traj in self.data:
      # plt.plot(oi[0, :], oi[1, :], linewidth=1)
      # for oi in obs:
        # plt.plot(oi, linewidth=1)
      plt.plot(obs[0], obs[1], linewidth=1)

    plt.tight_layout()
    saveto = os.path.join(self.path, 'content/map_gen.png')
    plt.savefig(saveto)
    plt.show(block = True)
        


if __name__ == "__main__":
  sim = Simulation(9)
  sim.generate_data()
  sim.save_data()
  sim.plot_data()
