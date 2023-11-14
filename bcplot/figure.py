import argparse
import os
import os.path as osp
import matplotlib.figure as figure
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from time import time

from .params import PARAMS


class Figure(object):

    def __init__(self, default_params=PARAMS, **kwargs):
        for key, value in default_params.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.parser = argparse.ArgumentParser()
        self.reset()

    def new_param(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def get_kwargs(self):
        return vars(self.parser.parse_args())

    def reset(self):
        self.start_time = time()
        self.__figure__()

    def __figure__(self):
        self.fig = figure.Figure(figsize=self.figsize, dpi=self.dpi)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot()
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_axis_off()
        self.__copyright__()

    def __copyright__(self, keys=['ratio', 'text', 'fname', 'size', 'fc', 'ec', 'lw']):
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