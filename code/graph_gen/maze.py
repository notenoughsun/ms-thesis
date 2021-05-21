import os.path
import inspect
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx

import math

# https://habr.com/ru/post/537630/


class maze(object):

    def __init__(self, boxsize, G=None, Z=None):
        self.boxsize = boxsize
        
        self.x = np.arange(0, boxsize, 1)
        self.y = np.arange(0, boxsize, 1)
        self.Z = np.zeros((boxsize, boxsize))

        if not G:
            self.G = nx.Graph()
        else:
            assert isinstance(G, nx.Graph)
            self.G = G
            self.Z = Z

        # self.im = None

    def generate(self):   
        Z = self.Z
        G = self.G
        boxsize = self.boxsize

        offsets_double = [[-2, 0], [2, 0], [0, -2], [0, 2]]
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def color(Z, ci):
            Z[ci[0]][ci[1]] = 1

        def add_valid(cp, active_set):
            for off in offsets_double:
                if cp[0] + off[0] >= 0 and cp[0] + off[0] < boxsize \
                and cp[1] + off[1] >= 0 and cp[1] + off[1] < boxsize:
                    if Z[cp[0] + off[0]][cp[1] + off[1]] == 0:
                        active_set.append([cp[0] + off[0], cp[1] + off[1]])

        def connect_to_existing(cp):
            def xytoname(ci):
                return ci[0] * boxsize + ci[1]
            off = np.random.randint(4)
            for i in range(4):
                id = (i + off) % 4
                cpx = cp[0] + offsets_double[id][0]
                cpy = cp[1] + offsets_double[id][1]
                if cpx >= 0 and cpx < boxsize \
                and cpy >= 0 and cpy < boxsize:
                    if Z[cpx][cpy] == 1:
                        cm = [cp[0] + offsets[id][0], cp[1] + offsets[id][1]]

                        if Z[cm[0], cm[1]] == 1:
                            return
                        # Z[cp[0] + offsets[id][0], cp[1] + offsets[id][1]] = 1 #color z
                        p1 = xytoname(cp)
                        p2 = xytoname(cm)
                        p3 = xytoname([cpx, cpy])
                        G.add_node(p1, pos = cp)
                        G.add_node(p2, pos = cm)
                        G.add_node(p3, pos = [cpx, cpy])
                        color(Z, cm)

                        G.add_edge(p1, p2, weight = 1.)
                        G.add_edge(p2, p3, weight = 1.)
                        return
            raise BaseException

        # cp = np.random.randint(self.boxsize / 2., size= 2)
        # cp = [cp[0] * 2, cp[1] * 2]
        active_set = []
        cp = [1, 1]
        color(Z, cp)
        add_valid(cp, active_set)

        while len(active_set) > 0:
            cp = active_set.pop(np.random.randint(len(active_set)))
            if Z[cp[0]][cp[1]] == 1:
                continue

            color(Z, cp)
            connect_to_existing(cp)
            connect_to_existing(cp)

            add_valid(cp, active_set) #active_set increased
        return G

    def plot(self, fig = None):
        if not fig:
            fig = plt.figure(num = "field2", figsize=(3,3), dpi = 150)
        self.im = plt.pcolormesh(
            self.x, self.y, self.Z, alpha=0.4, shading='auto')

    # def check(self, x, y):
    #     x_ = math.floor(x)
    #     y_ = math.floor(y)
    #     return self.Z[x_, y_]



filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

if __name__ == "__main__":
    m = maze(boxsize = 11)
    G = m.generate()
    m.plot()
    plt.axis('off')

    plt.savefig(os.path.join(path, "content/maze.png"))
    # plt.show(block = True)
    plt.show()

    Z = m.Z
    print(Z)
