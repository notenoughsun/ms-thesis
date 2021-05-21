import json
import os.path
import inspect
import networkx as nx

from networkx.readwrite.json_graph import jit_data, jit_graph
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_pydot import write_dot, read_dot
# nx.drawing.nx_pydot.write_dot

from maze import maze

# from networkx.drawing.nx_pydot import write_dot
# from scipy.interpolate import Rbf

# pos = nx.nx_agraph.graphviz_layout(G)
# pos=nx.spring_layout(G)
# print(G.nodes.data)

class Mgraph(maze):
    '''
    the map graph contains info about locations and can be plotted, 
    we add the method for tracing routes for map coverage
    next step:: implement the map
    '''
    def __init__(self, boxsize, G=None, Z=None):
        super().__init__(boxsize, G, Z)

        if G == None:
            self.generate()

        # dd = dict(self.G.nodes.data())
        self.nodes = self.G.nodes()
        # self.pos = {i : dd[i]['pos'] for i in list(dd.keys()) }
        self.pos = dict(self.G.nodes.data('pos'))

    def plot_field(self, fig = None, alpha = 0.4):
        if not fig:
            fig = plt.figure(num = "field2", figsize=(3,3), dpi = 150)
        # self.im = plt.pcolormesh(self.x, self.y, self.Z, alpha=0.4, shading='auto')
        offx, offy = 0, 0
        x = np.arange(offx, self.boxsize + offy, 1)
        y = np.arange(offx, self.boxsize + offy, 1)
        plt.pcolormesh(y, x, self.Z.T, alpha = 0.8, shading='auto')
        # plt.pcolormesh(x, y, self.Z, alpha=0.4, shading='auto')
        plt.tight_layout()
        plt.axis('off')

    def draw_graph(self, alpha = 0.6, node_size = 5, width = 2):
        nx.draw(self.G, pos = self.pos, alpha = alpha, node_size = node_size, width = width)

    def gen_routes(self, routes = None):
        """
        generate trajectories between random points (a, b) using dijkstra algorithm
        """
        G = self.G
        nodes = self.nodes
        pos = self.pos

        datasetgt = []  
        trajs = [[]]

        for route in routes:
            ids = np.take(nodes, route.squeeze())

            traj = [ids[0]] #initial point
            for i in range(len(ids) - 1):
                traj += (nx.dijkstra_path(G, ids[i], ids[i+1])[1:]) #update for circular routes
                
            trajs.append(traj)
                # H = G.subgraph(traj)
                # nx.draw_networkx_edges(H, pos = pos, edge_color='g', width = 6, alpha=0.2)                
            coords = np.array([pos[ti] for ti in traj])
            datasetgt.append(coords)

        # ids = np.take(nodes, routes.squeeze()).reshape(-1, 2)
        # for idi in ids:
        #     traj = nx.dijkstra_path(G, idi[0], idi[1])
        #     trajs.append(traj)
        #     # H = G.subgraph(traj)
        #     # nx.draw_networkx_edges(H, pos = pos, edge_color='g', width = 6, alpha=0.2)
            
        #     coords = np.array([pos[ti] for ti in traj])
        #     datasetgt.append(coords)
        return datasetgt, trajs

if __name__ == "__main__":
    mg = Mgraph(9)
    G = mg.G

    data = jit_data(G)

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    saveto = os.path.join(path, 'content/file.json')
    with open(saveto, 'w') as file:
        # file.write(data)
        json.dump(data, file)
    # write_dot(G, '/home/tim/git/thesis/code/graph_gen' + '/content/file.dot')

    # with open(saveto, "r") as read_file:
    #     data_read = json.load(read_file)

    # assert hash(data_read) == hash(data)
    # print("read write completed")

    n = 2 * len(mg.nodes)
    selected = np.random.randint(0, len(mg.nodes), size= n)
    routes =  np.array_split(selected, n // 4)  
    datasetgt, trajs = mg.gen_routes(routes)

    fig = plt.figure(num = "field3", figsize=(3,3), dpi = 150)
    mg.plot_field(fig)
    nx.draw(G, pos = mg.pos, alpha = 0.6, node_size = 5, width = 2)
    for ti in trajs:
        H = G.subgraph(ti)
        nx.draw_networkx_edges(H, pos = mg.pos, edge_color='g', width = 6, alpha=0.2)
    
    plt.savefig(os.path.join(path, 'content/routes.png'))
    plt.show()

    # uploading is not working
    
    # G2 = nx.Graph(read_dot('/home/tim/git/thesis/code/graph_gen' + '/content/file.dot'))
    # mg2 = Mgraph(G2, mg.Z)
    # datasetgt2, trajs = mg2.gen_routes(routes)
    # assert (datasetgt2 == datasetgt).all()
