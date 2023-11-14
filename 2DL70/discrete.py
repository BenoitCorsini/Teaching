import sys
import numpy as np
import numpy.random as npr

sys.path.append('../')
from bcplot import BCPlot as plot

CMAP = plot.get_cmap([
    (255/255, 255/255, 255/255),
    ( 64/255, 149/255, 191/255),
    (  0/255,   0/255,   0/255),
])
PARAMS = {
}


class DiscretePlot(plot):

    def __init__(self):
        super().__init__(**PARAMS)

    def file_name(self):
        return 'discrete'

    def image(self):
        self.reset()
        self.save_image(name=self.file_name())

    def video(self):
        self.reset()
        self.save_video(name=self.file_name())

    def run(self, seed=None):
        npr.seed(seed)
        self.image()
        # self.video()


if __name__ == '__main__':
    DP = DiscretePlot()
    DP.new_param('--seed', type=int, default=None)
    DP.run()