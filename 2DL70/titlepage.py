import sys
import numpy as np
import numpy.random as npr
from scipy.spatial import Delaunay

sys.path.append('../')
from bcplot import BCPlot as plot


PARAMS = {
    'copyright' : {},
    'figsize' : (16,9),
    'dpi' : 1000,
    'xmax' : 16,
    'ymax' : 9,
    'extra_perc' : 0.2,
    'lw' : 0.75,
    'ec' : plot.rgb_to_hex(204, 238, 255),
    'cmin' : plot.rgb_to_hex(255, 255, 255),
    'cmax' : plot.rgb_to_hex(229, 246, 255),
}


class TitlePage(plot):

    def __init__(self):
        super().__init__(**PARAMS)
        self.cmap = self.get_cmap([self.cmin, self.cmax])

    def triangulation(self, n_points):
        self.points = npr.rand(n_points, 2)
        self.points[:,0] = (self.xmax - self.xmin)*((1 + self.extra_perc)*self.points[:,0] - self.extra_perc/2)
        self.points[:,1] = (self.ymax - self.ymin)*((1 + self.extra_perc)*self.points[:,1] - self.extra_perc/2)
        self.simplices = Delaunay(self.points).simplices

    def create(self, n_points, seed=None):
        npr.seed(seed)
        self.triangulation(n_points)
        for simplex in self.simplices:
            self.plot_shape(
                shape_name='Polygon',
                xy=self.points[simplex,:],
                lw=self.lw,
                ec=self.ec,
                fc=self.cmap(npr.rand()),
            )
        self.save_image('titlepage')


if __name__ == '__main__':
    TP = TitlePage()
    TP.new_param('--seed', type=int, default=None)
    TP.new_param('--n_points', type=int, default=1000)
    TP.create(**TP.get_kwargs())