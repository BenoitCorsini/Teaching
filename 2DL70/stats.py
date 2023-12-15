import sys
import os.path as osp
import numpy as np
import pandas as pd
from scipy.special import gamma

sys.path.append('../')
from bcplot import BCPlot as plot

CMAP = plot.get_cmap([
    (255/255, 255/255, 255/255),
    ( 64/255, 149/255, 191/255),
    (  0/255,   0/255,   0/255),
])
PARAMS = {
    # 'dpi' : 500,
    'extra_left' : 0.2,
    'extra_right' : 0.2,
    'extra_bottom' : 0.1,
    'extra_top' : 0.25,
    'max_ticks' : 15,
    'tick_height' : 0.007,
    'label_xshift' : 0.007,
    'label_yshift' : 0.007,
    'label_height' : 0.01,
    'boxplot_ratio' : 0.5,
    'boxplot_bar_height' : 0.2,
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
    'boxplot_params' : {
        'lw' : 2,
        'ec' : CMAP(0.5),
        'fc' : CMAP(0.4),
        'zorder' : 0,
        'capstyle' : 'round',
        'joinstyle' : 'round',
    },
    'evo_params' : {
        'lw' : 2,
        'color' : CMAP(0.75),
        'zorder' : 3,
        'capstyle' : 'round',
        'joinstyle' : 'round',
        'fill' : False,
        'closed' : False,
    },
    'fluct_params' : {
        'lw' : 2,
        'color' : CMAP(0.25),
        'zorder' : 1,
        'capstyle' : 'round',
        'joinstyle' : 'round',
        'fill' : False,
        'closed' : False,
    },
    'times' : {
        'initial' : 1,
        'final' : 1,
        'steps' : 8,
    },
}



class Dataset(object):

    def __init__(self, data_file, data_dir='../data'):
        self.data_dir = data_dir
        self.data_file = data_file
        self.file = osp.join(self.data_dir, self.data_file)

    def get_xbounds(self):
        vmin = np.min(self.data)
        vmax = np.max(self.data)
        return vmin, vmax

    def get_counts(self, bars, xticks, normalize):
        tick_diff = xticks[1] - xticks[0]
        assert np.all(xticks[1:] - xticks[:-1] == tick_diff)
        if bars >= 1:
            bars = int(bars)
            ticks = np.stack([xticks + i*tick_diff/bars for i in range(bars)])
            ticks = np.reshape(ticks, -1, order='F')
        else:
            ticks = xticks[np.arange(0, np.size(xticks), step=int(1/bars))]
            ticks = np.concatenate([ticks, [ticks[-1] + int(1/bars)*tick_diff]])
        tick_diff = ticks[1] - ticks[0]
        assert np.all(np.abs(ticks[1:] - ticks[:-1] - tick_diff) < 1e-10)
        counts = np.array([np.sum((bottom <= self.data)*(top >= self.data)) for (bottom, top) in zip(ticks[:-1], ticks[1:])], dtype=int)
        if normalize:
            counts = counts/np.size(self.data)
        return ticks, counts

    def raw(self):
        data = self.data[np.random.permutation(np.size(self.data))]
        s = ''
        for entry in data:
            s += f'{entry}, '
        s = s[:-2]
        print(s)

    def print(self):
        mean = np.sum(self.data)/np.size(self.data)
        print(f'Mean: {mean}')
        variance = np.sum((self.data - mean)**2)/(np.size(self.data) - 1)
        print(f'Sample Variance: {variance}')
        print(f'Sample STD: {np.sqrt(variance)}')
        variance = np.sum((self.data - mean)**2)/np.size(self.data)
        print(f'Population Variance: {variance}')
        print(f'Population STD: {np.sqrt(variance)}')
        print(f'Range: {np.max(self.data) - np.min(self.data)}')
        print(f'Median: {np.quantile(self.data, [0.5])}')
        print(f'Quartiles: {np.quantile(self.data, [0.25, 0.75])}')
        print(f'Quantiles: {np.quantile(self.data, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])}')
        print()
        threshold = 643801
        # print(f'Proportion below: {100*np.sum(self.data <= threshold)/np.size(self.data):0.2f}%')
        print(f'Proportion between: {100*np.mean((self.data <= 17.099)*(7.201 <= self.data))}')

class Temperature(Dataset):

    def __init__(self, temperature_file='Eindhoven 2022.csv'):
        super().__init__(data_file=temperature_file)
        self.temperatures = pd.read_csv(self.file, index_col=0)
        self.Tmin = self.temperatures['Min Temperature'].to_numpy()
        self.Tavg = self.temperatures['Avg Temperature'].to_numpy()
        self.Tmax = self.temperatures['Max Temperature'].to_numpy()
        # self.data = np.sort(np.concatenate([self.Tmin, self.Tavg, self.Tmax]))
        assert np.all(self.Tmin <= self.Tavg)
        assert np.all(self.Tmax >= self.Tavg)
        self.data = self.Tavg#[np.random.permutation(np.size(self.Tavg))]

class Country(Dataset):

    def __init__(self, country_file='country.csv'):
        super().__init__(data_file=country_file)
        self.country = pd.read_csv(self.file, index_col=0)
        self.pop = self.country['Population'].to_numpy()
        self.pop = np.array([pop.replace(',', '') for pop in self.pop]).astype(int)
        self.size = self.country['Size'].to_numpy()
        self.size = np.array([size.replace(',', '') for size in self.size]).astype(float)
        self.data = self.pop
        self.data = self.size
        self.data = np.log(1 + self.pop)
        self.data = np.log(1 + self.size)

class Trains(Dataset):

    def __init__(self, train_file='train.csv'):
        super().__init__(data_file=train_file)
        self.train = pd.read_csv(self.file, index_col=0)
        self.data = self.train['time'].to_numpy()
        assert np.all(self.data >= 0)
        assert np.all(self.data < 60)
        assert np.all(self.data[1:] > self.data[:-1])
        self.data = np.concatenate([self.data, [60 + self.data[0]]])
        self.data = self.data[1:] - self.data[:-1]


class StatsPlot(plot):

    def __init__(self):
        super().__init__(**PARAMS)

    def file_name(self, data):
        return data.__class__.__name__.lower()

    def histogram_setup(self, data, bars=1, normalize=False):
        xticks = self.get_ticks(
            bounds=data.get_xbounds(),
            extras=(self.extra_left, self.extra_right),
            axis='x',
        ).astype(int)
        histo_info = data.get_counts(bars, xticks, normalize)
        if normalize:
            yticks = self.get_ticks(
                bounds=(0, np.max(histo_info[1])),
                extras=(self.extra_bottom, self.extra_top),
                axis='y',
                step=0.01,
            )
        else:
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
            - np.arange(step, - minimum + 1 + step, step=step)[::-1],
            np.arange(0, maximum + 1 + step, step=step),
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

    def plot_bars(self, ticks, counts):
        for bottom, top, height in zip(ticks[:-1], ticks[1:], counts):
            self.plot_shape(
                shape_name='Rectangle',
                xy=(bottom, 0),
                width=top - bottom,
                height=height,
                **self.histo_params
            )

    def plot_normal(self, data, norm, n_points=200):
        mean = np.mean(data.data)
        std = np.std(data.data)
        x = np.arange(n_points + 1)/n_points
        x = self.xmin + (self.xmax - self.xmin)*x
        y = norm*np.exp( - (x - mean)**2/2/std**2)/std/np.sqrt(2*np.pi)
        self.plot_shape(
            shape_name='Polygon',
            xy=np.stack([x, y], axis=-1),
            **self.normal_params
        )

    def plot_beta(self, data, norm, n_points=200):
        mean = np.mean(data.data)
        std = np.std(data.data)
        b = np.min(data.data)
        a = np.max(data.data) - b
        m = (mean - b)/a
        v = std**2/a**2
        alpha = (m*(1 - m)/v - 1)*m
        beta = (m*(1 - m)/v - 1)*(1 - m)
        x = np.arange(n_points + 1)/n_points
        y = x**(alpha - 1)*(1 - x)**(beta - 1)
        y *= gamma(alpha + beta)/gamma(alpha)/gamma(beta)
        y = norm*np.concatenate([[0], y, [0]])*a/n_points
        x = a*x + b
        x = np.concatenate([[self.xmin], x, [self.xmax]])
        self.plot_shape(
            shape_name='Polygon',
            xy=np.stack([x, y], axis=-1),
            **self.normal_params
        )

    def plot_exponential(self, data, l, norm, n_points=200):
        x = np.arange(n_points + 1)/n_points
        x = self.xmin + (self.xmax - self.xmin)*x
        index = np.sum(x <= 0)
        x = np.concatenate([x[:index], [0]*2, x[index:]])
        y = norm*l*np.exp( - l*x)
        y[:index + 1] = 0
        self.plot_shape(
            shape_name='Polygon',
            xy=np.stack([x, y], axis=-1),
            **self.normal_params
        )

    def histogram(self, data, bars=1, transparent=False, normalize=False, **kwargs):
        ticks, counts = self.histogram_setup(data, bars, normalize)
        self.plot_bars(ticks, counts)
        # self.plot_normal(data, np.sum(counts)*(ticks[1] - ticks[0]))
        # self.plot_beta(data, np.sum(counts)*(ticks[1] - ticks[0]))
        print(data.data)
        l = np.log(4/3)/np.quantile(data.data, 0.25)
        l = 1/np.mean(data.data)
        l = np.mean(1/data.data)
        print(l)
        # self.plot_exponential(data, l, np.sum(counts)*(ticks[1] - ticks[0]))
        self.save_image(name=self.file_name(data), transparent=transparent)

    def boxplot_setup(self, data, box_width):
        quartile1, median, quartile3 = np.quantile(data.data, [0.25, 0.5, 0.75])
        iqr = quartile3 - quartile1
        self.xmin = median - iqr/box_width/2
        self.xmax = median + iqr/box_width/2
        self.ymin = 1/box_width/self.boxplot_ratio
        self.ymax = - 1/box_width/self.boxplot_ratio
        self.x_over_y = (self.xmax - self.xmin)/(self.ymax - self.ymin)*self.figsize[1]/self.figsize[0]
        self.reset()
        return median, (quartile1, quartile3)

    def plot_quantiles(self, data, median, quartiles):
        iqr = quartiles[1] - quartiles[0]
        lower = np.min(data.data[data.data >= quartiles[0] - 1.5*iqr])
        upper = np.max(data.data[data.data <= quartiles[1] + 1.5*iqr])
        outliers = data.data.copy()
        outliers = outliers[(outliers < lower) + (outliers > upper)]
        extremes = outliers[(outliers < lower - 1.5*iqr) + (outliers > upper + 1.5*iqr)]
        outliers = outliers[(outliers >= lower - 1.5*iqr) & (outliers <= upper + 1.5*iqr)]
        self.plot_shape(
            shape_name='Rectangle',
            xy=(lower, 0),
            width=upper - lower,
            height=0,
            **self.boxplot_params
        )
        self.plot_shape(
            shape_name='Rectangle',
            xy=(lower, - self.boxplot_bar_height),
            width=0,
            height=2*self.boxplot_bar_height,
            **self.boxplot_params
        )
        self.plot_shape(
            shape_name='Rectangle',
            xy=(upper, - self.boxplot_bar_height),
            width=0,
            height=2*self.boxplot_bar_height,
            **self.boxplot_params
        )
        self.plot_shape(
            shape_name='Rectangle',
            xy=(quartiles[0], - 1),
            width=iqr,
            height=2,
            **self.boxplot_params
        )
        self.plot_shape(
            shape_name='Rectangle',
            xy=(median, - 1),
            width=0,
            height=2,
            **self.boxplot_params
        )
        for o in outliers:
            self.plot_shape(
                shape_name='Ellipse',
                xy=(o, 0),
                width=2*self.boxplot_bar_height*self.x_over_y,
                height=2*self.boxplot_bar_height,
                fill=False,
                **self.boxplot_params
            )
        for e in extremes:
            self.plot_shape(
                shape_name='Polygon',
                xy=[
                    (e - self.boxplot_bar_height*self.x_over_y, - self.boxplot_bar_height),
                    (e + self.boxplot_bar_height*self.x_over_y, self.boxplot_bar_height),
                ],
                **self.boxplot_params
            )
            self.plot_shape(
                shape_name='Polygon',
                xy=[
                    (e - self.boxplot_bar_height*self.x_over_y, self.boxplot_bar_height),
                    (e + self.boxplot_bar_height*self.x_over_y, - self.boxplot_bar_height),
                ],
                **self.boxplot_params
            )

    def boxplot(self, data, box_width=1, transparent=False, **kwargs):
        boxplot_info = self.boxplot_setup(data, box_width)
        self.plot_quantiles(data, *boxplot_info)
        self.save_image(name=self.file_name(data), transparent=transparent)

    def evolution_setup(self, data):
        xticks = self.get_ticks(
            bounds=(1, np.size(data.data)),
            extras=(self.extra_left, self.extra_right),
            axis='x',
        ).astype(int)
        bounds = (
            np.floor(np.min(data.data)),
            np.ceil(np.max(data.data)),
        )
        yticks = self.get_ticks(
            bounds=bounds,
            extras=(self.extra_bottom, self.extra_top),
            axis='y',
            step=1,
        ).astype(int)
        self.x_over_y = (self.xmax - self.xmin)/(self.ymax - self.ymin)*self.figsize[1]/self.figsize[0]
        self.reset()
        self.plot_axis(xticks, yticks)

    def evolution(self, data, transparent=False, **kwargs):
        self.evolution_setup(data)

        self.evo = self.plot_shape(
            shape_name='Polygon',
            xy=[[0, 0]],
            **self.evo_params
        )
        evo = np.array(list(enumerate(data.data)))
        evo[:,0] += 1
        evo = np.concatenate([[[0, 0]], evo])
        self.flucts = []
        for _ in range(3):
            self.flucts.append(self.plot_shape(
                shape_name='Polygon',
                xy=[[0, 0]],
                **self.fluct_params
            ))
        flucts = []
        for q in [0.25, 0.5, 0.75]:
            flucts.append([0] + [np.quantile(data.data[:(i + 1)], q) for i in range(np.size(data.data))])

        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        steps = np.arange(0, self.times['steps'] + 1/self.fps, 1/self.fps)/self.times['steps']
        assert np.max(steps) == 1
        steps = (np.size(data.data)*steps).astype(int)
        for s in steps:
            self.evo.set_xy(evo[:(s + 1)])
            for p, f in zip(self.flucts, flucts):
                p.set_xy(np.array(list(enumerate(f[:(s + 1)]))))
            self.new_frame()
        for _ in range(int(self.fps*self.times['final'])):
            self.new_frame()

        self.save_image(name=self.file_name(data), transparent=transparent)
        self.save_video(name=self.file_name(data))



if __name__ == '__main__':
    Data = Temperature()
    # Data = Country()
    # Data = Trains()
    # Data.raw()
    Data.print()
    SP = StatsPlot()
    SP.new_param('--transparent', type=int, default=0)
    SP.new_param('--bars', type=float, default=1)
    SP.new_param('--normalize', type=int, default=0)
    SP.new_param('--box_width', type=float, default=1)
    # SP.histogram(Data, **SP.get_kwargs())
    # SP.boxplot(Data, **SP.get_kwargs())
    # SP.evolution(Data, **SP.get_kwargs())