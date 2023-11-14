import os
import os.path as osp
import itertools
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from time import time

from .params import PARAMS


class Image(object):

    def __init__(self, default_params=PARAMS, **kwargs):
        self.start_time = time()
        for key, value in default_params.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__figure__()
        self.__images__()

    def __figure__(self):
        self.fig = figure.Figure(figsize=self.figsize, dpi=self.dpi)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        bound = 1 + self.extra_space
        self.ax = self.fig.add_subplot()
        self.ax.set_xlim(-bound, bound)
        self.ax.set_ylim(-bound, bound)
        self.ax.set_axis_off()

        self.frame = patches.Polygon([[-1,-1], [1,-1], [1,1], [-1,1]])
        self.ax.add_patch(self.frame)
        self.__copyright__(bound)

    def __copyright__(self, bound=1, keys=['ratio', 'text', 'fname', 'size', 'fc', 'ec', 'lw']):
        kwargs = {key:value for (key,value) in self.copyright.items() if key not in keys}
        prop = FontProperties(fname=osp.join(osp.dirname(osp.realpath(__file__)), self.copyright['fname']))
        path = TextPath(
            (-self.copyright['ratio']*bound, -self.copyright['ratio']*bound),
            self.copyright['text'],
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

    def __images__(self):
        self.nx, self.ny = self.figsize
        self.nx = int(self.nx*self.dpi)
        self.ny = int(self.ny*self.dpi)
        self.im = np.empty((self.nx,self.ny,4))
        self.extent = (-1-self.extra_space,1+self.extra_space,-1-self.extra_space,1+self.extra_space)
        self.grid_x = np.arange(self.nx).reshape((1,self.nx)).repeat(self.ny,axis=0)
        self.grid_x = (2*self.grid_x/self.nx - 1)*(1 + self.extra_space)
        self.grid_y = np.arange(self.ny).reshape((self.ny,1)).repeat(self.nx,axis=1)
        self.grid_y = (2*self.grid_y/self.ny - 1)*(1 + self.extra_space)

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

    def reset(self):
        self.start_time = time()
        self.__figure__()

    def time(self):
        return self.time_to_string(time() - self.start_time)

    def grid(self, n_blocks=10, **kwargs):
        columns = []
        rows = []
        kwargs['capstyle'] = 'butt'
        kwargs['clip_path'] = self.frame
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

    def pointer(self, **kwargs):
        pointer = [
            patches.Rectangle(
                xy=(-1,-1),
                width=2,
                height=2,
                clip_path=self.frame,
                **kwargs
            ),
            patches.Rectangle(
                xy=(-1,-1),
                width=2,
                height=2,
                clip_path=self.frame,
                **kwargs
            ),
        ]
        for p in pointer:
            self.ax.add_patch(p)
        return pointer

    def text(self, text='', pos=(0,0), **kwargs):
        return self.ax.text(pos[0], pos[1], text, **kwargs)

    def plot_shape(self, shape_name, **kwargs):
        patch = getattr(patches, shape_name)(**kwargs)
        self.ax.add_patch(patch)
        return patch

    def combine_paths(self, paths, **kwargs):
        patch = path.Path.make_compound_path(*[path.Path(vertices=P) for P in paths])
        patch = patches.PathPatch(patch, **kwargs)
        self.ax.add_patch(patch)
        return patch

    def im_fades(self, xy=(0,0), radius=1, color='white', scale=lambda x:x, **kwargs):
        im = self.im.copy()
        im[:,:,0], im[:,:,1], im[:,:,2] = to_rgb(color)
        dist = np.sqrt((self.grid_x - xy[0])**2 + (self.grid_y - xy[1])**2)
        dist = 1 - dist/radius
        dist = dist*(dist > 0)
        dist = scale(dist)
        im[:,:,3] = dist
        return im

    def plot_fade(self, xy=(0,0), radius=1, color='white', scale=lambda x:x, **kwargs):
        im = self.im_fades(xy, radius, color)
        return self.ax.imshow(im, vmin=0, vmax=1, origin='lower', extent=self.extent, **kwargs)

    def line(self, X, Y, **kwargs):
        line = path.Path(np.array([X, Y]).T)
        line = patches.PathPatch(line, **kwargs)
        self.ax.add_patch(line)
        return line

    def pattern(self, patch, density=10, delta=0, marker='.', **kwargs):
        bbox = patch.get_extents()
        x0 = 2*bbox.x0/self.nx - 1
        x1 = 2*bbox.x1/self.nx - 1
        y0 = 2*bbox.y0/self.ny - 1
        y1 = 2*bbox.y1/self.ny - 1
        sx = int((x1 - x0)*density)
        sy = int((y1 - y0)*density)
        x, y = zip(*itertools.product(range(sx), range(sy)))
        x = (2*np.array(x) + 1)*(x1 - x0)/sx/2 + x0 + delta*(2*npr.rand(sx*sy) - 1)
        y = (2*np.array(y) + 1)*(y1 - y0)/sy/2 + y0 + delta*(2*npr.rand(sx*sy) - 1)
        return self.ax.plot(x, y, lw=0, marker=marker, **kwargs)
