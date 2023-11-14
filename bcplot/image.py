import os
import os.path as osp
import itertools
import numpy as np
import matplotlib.figure as figure
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from time import time

from .params import PARAMS


class Image(object):

    def __init__(self, default_params=PARAMS, **kwargs):
        for key, value in default_params.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.reset()

    def reset(self):
        self.start_time = time()
        self.figure()

    def figure(self):
        self.fig = figure.Figure(figsize=self.figsize, dpi=self.dpi)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.ax = self.fig.add_subplot()
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_axis_off()
        self.copyright()

    def copyright(self, keys=['ratio', 'text', 'fname', 'size', 'fc', 'ec', 'lw']):
        kwargs = {key:value for (key,value) in self.copyright.items() if key not in keys}
        prop = FontProperties(fname=osp.join(osp.dirname(osp.realpath(__file__)), self.copyright['fname']))
        path = TextPath(
            xy=(
                self.xmin + self.copyright['ratio']*(self.xmax - self.xmin),
                self.ymin + self.copyright['ratio']*(self.ymax - self.ymin),
            ),
            s=self.copyright['text'],
            prop=prop,
            size=self.copyright['size']
        )
        self.ax.add_patch(patches.PathPatch(
            path=path,
            color=self.copyright['ec'],
            lw=self.copyright['lw'],
            **kwargs
        ))
        self.ax.add_patch(patches.PathPatch(
            path=path,
            color=self.copyright['fc'],
            **kwargs
        ))

    def save(self, name='image', image_dir=None, transparent=False):
        if image_dir is None:
            image_dir = self.image_dir
        if not osp.exists(image_dir):
            os.makedirs(image_dir)
        self.fig.savefig(osp.join(image_dir, name + '.png'), transparent=transparent)

    @staticmethod
    def get_cmap(colour_list):
        return LinearSegmentedColormap.from_list(f'cmap of Benoit', colour_list)

    @staticmethod
    def get_greyscale(start_with_white=True):
        if start_with_white:
            colour_list = ['white', 'black']
        else:
            colour_list = ['black', 'white']
        return LinearSegmentedColormap.from_list(f'greyscale of Benoit', colour_list)

    @staticmethod
    def get_cmap_from_colour(colour='grey', start_with='white', end_with='black'):
        if (start_with == 'same') & (end_with == 'same'):
            raise UserWarning(f'The cmap is uniformly coloured!')
            colour_list = [colour]*2
        elif start_with == 'same':
            colour_list = [colour, end_with]
        elif end_with == 'same':
            colour_list = [start_with, colour]
        else:
            colour_list = [start_with, colour, end_with]

        return Image.get_cmap(colour_list)

    @staticmethod
    def time_to_string(time):
        s = ''
        hours = int(time/3600)
        minutes = int((time - 60*hours)/60)
        seconds = int(time - 3600*hours - 60*minutes)
        if hours:
            s = f'{hours}h{minutes}m{seconds}s'
        elif minutes:
            s = f'{minutes}m{seconds}s'
        else:
            s = f'{seconds}s'
        return s

    def time(self):
        return self.time_to_string(time() - self.start_time)

    def plot_shape(self, shape_name, **kwargs):
        patch = getattr(patches, shape_name)(**kwargs)
        self.ax.add_patch(patch)
        return patch

    def grid(self, n_blocks=10, **kwargs):
        columns = []
        rows = []
        kwargs['capstyle'] = 'butt'
        delta = 2/n_blocks
        for i in range(0, n_blocks+1):
            pos = i*delta - 1
            column_patch = patches.Polygon(
                [[pos,-1], [pos,1]],
                **kwargs
            )
            self.ax.add_patch(column_patch)
            columns.append(column_patch)
            row_patch = patches.Polygon(
                [[-1,pos], [1,pos]],
                **kwargs
            )
            self.ax.add_patch(row_patch)
            rows.append(row_patch)
        return columns, rows