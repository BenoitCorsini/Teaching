import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap as LSC
from scipy.spatial import Delaunay


def rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'

PARAMS = {
    'figsize' : (16,9),
    'dpi' : 1000,
    'N' : 1000,
    'extra_perc' : 0.2,
    'lw' : 0.75,
    'ec' : rgb_to_hex(204, 238, 255),
    'cmin' : rgb_to_hex(255, 255, 255),
    'cmax' : rgb_to_hex(229, 246, 255),
}

class Teaxte(object):

    def __init__(self, **kwargs):
        for key, value in PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.nx, self.ny = self.figsize
        self.cmap = LSC.from_list('cmap', [self.cmin, self.cmax])
        self.__figure__()

    def __figure__(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_axis_off()
        self.ax.set_xlim(0, self.nx)
        self.ax.set_ylim(0, self.ny)

    def triangulation(self):
        self.points = npr.rand(self.N, 2)
        self.points[:,0] = (1 + self.extra_perc)*self.figsize[0]*self.points[:,0] - self.extra_perc*self.figsize[0]/2
        self.points[:,1] = (1 + self.extra_perc)*self.figsize[1]*self.points[:,1] - self.extra_perc*self.figsize[1]/2
        self.simplices = Delaunay(self.points).simplices

    def save(self, name='image'):
        self.triangulation()
        for simplex in self.simplices:
            self.ax.add_patch(Polygon(
                xy=self.points[simplex,:],
                lw=self.lw,
                ec=self.ec,
                fc=self.cmap(npr.rand()),
            ))
        self.fig.savefig(name + '.png')

    def titlepage(self):
        self.save('titlepage')


if __name__ == '__main__':
    T = Teaxte()
    T.titlepage()