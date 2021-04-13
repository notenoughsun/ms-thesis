import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx

from maze import maze

# from networkx.drawing.nx_pydot import write_dot
# from scipy.interpolate import Rbf

# pos = nx.nx_agraph.graphviz_layout(G)
# pos=nx.spring_layout(G)
# print(G.nodes.data)

class Mgraph(maze):
    def __init__(self):
        super().__init__()
        self.generate()

        dd = dict(self.G.nodes.data())
        self.nodes = self.G.nodes()
        self.pos = {i : dd[i]['pos'] for i in list(dd.keys()) }

    def plot_field(self, fig = None):
        if not fig:
            fig = plt.figure(num = "field2", figsize=(3,3), dpi = 150)
        # self.im = plt.pcolormesh(self.x, self.y, self.Z, alpha=0.4, shading='auto')
        offx, offy = 0, 0
        x = np.arange(offx, self.boxsize + offy, 1)
        y = np.arange(offx, self.boxsize + offy, 1)
        plt.pcolormesh(y, x, self.Z.T, alpha=0.4, shading='auto')
        # plt.pcolormesh(x, y, self.Z, alpha=0.4, shading='auto')
        plt.tight_layout()

    def gen_routes(self, n = 40):
        """
        generate trajectories between random points (a, b) using dijkstra algorithm
        """
        G = self.G
        nodes = self.nodes
        pos = self.pos

        datasetgt = []
        trajs = []

        # df = pd.DataFrame(trajs, columns=['traj', 'poses'])

        routes = np.random.randint(len(nodes), size = (2, n))
        ids = np.take(nodes, routes.squeeze()).reshape(-1, 2)
        for idi in ids:
            traj = nx.dijkstra_path(G, idi[0], idi[1])
            trajs.append(traj)
            # H = G.subgraph(traj)
            # nx.draw_networkx_edges(H, pos = pos, edge_color='g', width = 6, alpha=0.2)
            
            coords = np.array([pos[ti] for ti in traj])
            datasetgt.append(coords)
        return datasetgt, trajs

if __name__ == "__main__":
    mg = Mgraph()
    G = mg.G

    fig = plt.figure(num = "field3", figsize=(3,3), dpi = 150)

    mg.plot_field(fig)
    nx.draw(G, pos = mg.pos, alpha = 0.6, node_size = 5, width = 2)
    datasetgt = []
    n = 40

    nodes = mg.nodes
    pos = mg.pos

    # # generate trajectories between random points (a, b) using dijkstra algorithm
    routes = np.random.randint(len(nodes), size = (2, n))

    ids = np.take(nodes, routes.squeeze()).reshape(-1, 2)
    for idi in ids:
        traj = nx.dijkstra_path(G, idi[0], idi[1])
        H = G.subgraph(traj)
        nx.draw_networkx_edges(H, pos = pos, edge_color='g', width = 6, alpha=0.2)
        
        coords = np.array([pos[ti] for ti in traj])
        datasetgt.append(coords)
        
    plt.show()
    # nx.write_dot(G, '/content/file.dot')