import sys
import os.path as osp
import numpy as np
import pandas as pd

sys.path.append('../')
from bcplot import BCPlot as plot

CMAP = plot.get_cmap([
    (255/255, 255/255, 255/255),
    ( 64/255, 149/255, 191/255),
    (  0/255,   0/255,   0/255),
])
PARAMS = {
    # 'dpi' : 10,
    'extra_left' : 0.1,
    'extra_right' : 0.1,
    'extra_bottom' : 0.05,
    'extra_top' : 0.1,
    'max_ticks' : 15,
    'tick_height' : 0.007,
    'label_xshift' : 0.007,
    'label_yshift' : 0.007,
    'label_height' : 0.01,
    'grid_params' : {
        'lw' : 1,
        'color' : CMAP(0.75),
        'zorder' : 2,
        'alpha' : 0.1,
    },
    'axis_params' : {
        'lw' : 2,
        'color' : CMAP(0.75),
        'zorder' : 3,
        'capstyle' : 'round',
    },
    'label_params' : {
        'lw' : 1,
        'color' : CMAP(0.9),
        'zorder' : 3,
        'capstyle' : 'round',
        'joinstyle' : 'round',
    },
    'histo_params' : {
        'lw' : 2,
        'ec' : CMAP(0.5),
        'fc' : CMAP(0.4),
        'zorder' : 0,
        'capstyle' : 'round',
        'joinstyle' : 'round',
    },
    'normal_params' : {
        'lw' : 5,
        'color' : CMAP(0.6),
        'zorder' : 1,
        'joinstyle' : 'round',
        'capstyle' : 'round',
        'fill' : False,
        'closed' : False,
    },
}


class Temperature(object):

    def __init__(self, data_dir='../data', temperature_file='Eindhoven 2022.csv'):
        self.data_dir = data_dir
        self.temperature_file = temperature_file
        self.file = osp.join(self.data_dir, self.temperature_file)
        self.temperatures = pd.read_csv(self.file, index_col=0)
        self.Tmin = self.temperatures['Min Temperature'].to_numpy()
        self.Tavg = self.temperatures['Avg Temperature'].to_numpy()
        self.Tmax = self.temperatures['Max Temperature'].to_numpy()
        self.Tall = np.sort(np.concatenate([self.Tmin, self.Tavg, self.Tmax]))
        assert np.all(self.Tmin <= self.Tavg)
        assert np.all(self.Tmax >= self.Tavg)

    def get_xbounds(self):
        Tmin = np.min(self.Tall)
        Tmax = np.max(self.Tall)
        return Tmin, Tmax

    def get_counts(self, bars, xticks):
        tick_diff = xticks[1] - xticks[0]
        assert np.all(xticks[1:] - xticks[:-1] == tick_diff)
        ticks = np.stack([xticks + i*tick_diff/bars for i in range(bars)])
        ticks = np.reshape(ticks, -1, order='F')
        tick_diff = ticks[1] - ticks[0]
        assert np.all(np.abs(ticks[1:] - ticks[:-1] - tick_diff) < 1e-10)
        counts = np.array([np.sum((bottom <= self.Tall)*(top >= self.Tall)) for (bottom, top) in zip(ticks[:-1], ticks[1:])], dtype=int)
        return ticks, counts


class StatsPlot(plot):

    def __init__(self):
        super().__init__(**PARAMS)

    def file_name(self, data):
        return data.__class__.__name__.lower()

    def setup(self, data, bars=1):
        xticks = self.get_ticks(
            bounds=data.get_xbounds(),
            extras=(self.extra_left, self.extra_right),
            axis='x',
        ).astype(int)
        histo_info = data.get_counts(bars, xticks)
        yticks = self.get_ticks(
            bounds=(0, np.max(histo_info[1])),
            extras=(self.extra_bottom, self.extra_top),
            axis='y',
        ).astype(int)
        self.x_over_y = (self.xmax - self.xmin)/(self.ymax - self.ymin)*self.figsize[1]/self.figsize[0]
        self.reset()
        self.plot_axis(xticks, yticks)
        return histo_info

    def get_ticks(self, bounds, extras, axis, step=1):
        minimum = (1 + extras[0])*bounds[0] - extras[0]*bounds[1]
        maximum = (1 + extras[1])*bounds[1] - extras[1]*bounds[0]
        step *= np.ceil((maximum - minimum)/step/self.max_ticks)
        ticks = np.concatenate([
            - np.arange(step, - int(minimum) + 1 + step, step=step)[::-1],
            np.arange(0, int(maximum) + 1 + step, step=step),
        ])
        ticks = ticks[(minimum - step <= ticks) & (ticks <= maximum + step)]
        setattr(self, axis + 'min', minimum)
        setattr(self, axis + 'max', maximum)
        return ticks

    def plot_ticks(self, xticks, yticks):
        self.plot_shape(
            shape_name='Rectangle',
            xy=(self.xmin, 0),
            width=self.xmax - self.xmin,
            height=0,
            **self.axis_params
        )
        self.plot_shape(
            shape_name='Rectangle',
            xy=(0, self.ymin),
            width=0,
            height=self.ymax - self.ymin,
            **self.axis_params
        )
        label = self.path_from_string(
            s=str(0),
            x= - self.label_xshift*(self.xmax - self.xmin),
            y= - self.label_yshift*(self.ymax - self.ymin),
            height=self.label_height*(self.ymax - self.ymin),
            anchor='north east',
        )
        self.plot_shape(
            shape_name='PathPatch',
            path=label,
            **self.label_params
        )
        for xt in xticks:
            if xt:
                self.plot_shape(
                    shape_name='Rectangle',
                    xy=(xt, - (self.ymax - self.ymin)*self.tick_height/2),
                    width=0,
                    height=(self.ymax - self.ymin)*self.tick_height,
                    **self.axis_params
                )
                label = self.path_from_string(
                    s=str(xt),
                    x=xt - (self.xmax - self.xmin)*self.label_xshift,
                    y= - (self.ymax - self.ymin)*self.label_yshift,
                    height=(self.ymax - self.ymin)*self.label_height,
                    anchor='north east',
                )
                self.plot_shape(
                    shape_name='PathPatch',
                    path=label,
                    **self.label_params
                )
        tick_width = self.tick_height*self.x_over_y
        for yt in yticks:
            if yt:
                self.plot_shape(
                    shape_name='Rectangle',
                    xy=(- self.x_over_y*(self.ymax - self.ymin)*self.tick_height/2, yt),
                    width=self.x_over_y*(self.ymax - self.ymin)*self.tick_height,
                    height=0,
                    **self.axis_params
                )
                label = self.path_from_string(
                    s=str(yt),
                    x= - (self.xmax - self.xmin)*self.label_xshift,
                    y=yt - (self.ymax - self.ymin)*self.label_yshift,
                    height=(self.ymax - self.ymin)*self.label_height,
                    anchor='north east',
                )
                self.plot_shape(
                    shape_name='PathPatch',
                    path=label,
                    **self.label_params
                )

    def plot_axis(self, xticks, yticks):
        self.grid(
            x_blocks=np.size(xticks) - 1,
            y_blocks=np.size(yticks) - 1,
            xmin=np.min(xticks),
            xmax=np.max(xticks),
            ymin=np.min(yticks),
            ymax=np.max(yticks),
            **self.grid_params
        )
        self.plot_ticks(xticks, yticks)

    def histogram(self, ticks, counts):
        for bottom, top, height in zip(ticks[:-1], ticks[1:], counts):
            self.plot_shape(
                shape_name='Rectangle',
                xy=(bottom, 0),
                width=top - bottom,
                height=height,
                **self.histo_params
            )

    def plot_normal(self, T, norm, n_points=200):
        mean = np.mean(T.Tall)
        std = np.std(T.Tall)
        x = np.arange(n_points + 1)/n_points
        x = self.xmin + (self.xmax - self.xmin)*x
        y = norm*np.exp( - (x - mean)**2/2/std**2)/std/np.sqrt(2*np.pi)
        self.plot_shape(
            shape_name='Polygon',
            xy=np.stack([x, y], axis=-1),
            **self.normal_params
        )

    def plot_histogram(self, data, bars=1, transparent=False, **kwargs):
        ticks, counts = self.setup(data, bars)
        self.histogram(ticks, counts)
        self.plot_normal(data, np.sum(counts)*(ticks[1] - ticks[0]))
        self.save_image(name=self.file_name(data), transparent=transparent)


if __name__ == '__main__':
    SP = StatsPlot()
    SP.new_param('--transparent', type=int, default=0)
    SP.new_param('--bars', type=int, default=1)
    SP.plot_histogram(Temperature(), **SP.get_kwargs())