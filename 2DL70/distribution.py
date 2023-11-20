import sys
import numpy as np
import numpy.random as npr
from scipy.special import factorial

sys.path.append('../')
from bcplot import BCPlot as plot

CMAP = plot.get_cmap([
    (255/255, 255/255, 255/255),
    ( 64/255, 149/255, 191/255),
    (  0/255,   0/255,   0/255),
])
PMF_COLOUR = CMAP(0.7)
CDF_COLOUR = CMAP(0.3)
FLU_COLOUR = CMAP(0.5)
PARAMS = {
    'extra_left' : 0.1,
    'extra_right' : 0.1,
    'extra_bottom' : 0.1,
    'extra_top' : 0.15,
    'max_ticks' : 15,
    'tick_height' : 0.01,
    'label_xshift' : 0.1,
    'label_yshift' : 0.01,
    'label_height' : 0.02,
    'mean_shift' : - 0.05,
    'mean_height' : 0.015,
    'std_mult' : 1.96,
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
    'pmf_bar_params' : {
        'lw' : 5,
        'color' : PMF_COLOUR,
        'zorder' : 1,
    },
    'pmf_marker_params' : {
        'lw' : 0,
        'color' : PMF_COLOUR,
        'marker' : 'o',
        'ms' : 10,
        'zorder' : 1,
    },
    'cdf_bar_params' : {
        'lw' : 4,
        'color' : CDF_COLOUR,
        'zorder' : 0,
    },
    'cdf_empty_marker_params' : {
        'lw' : 0,
        'color' : CMAP(0.),
        'marker' : 'o',
        'ms' : 3,
        'zorder' : 0,
    },
    'cdf_marker_params' : {
        'lw' : 0,
        'color' : CDF_COLOUR,
        'marker' : 'o',
        'ms' : 8,
        'zorder' : 0,
    },
    'fluctuation_params' : {
        'color' : FLU_COLOUR,
        'lw' : 3,
        'zorder' : 0,
        'capstyle' : 'round',
    },
    'fluctuation_ends_params' : {
        'lw' : 0,
        'color' : FLU_COLOUR,
        'ms' : 10,
        'zorder' : 0,
    },
    'text_params' : {
        'x' : 0 + 0.2,
        'y' : 1 + 0.01,
        'anchor' : 'south west',
        'height' : 0.05,
    },
    'name_params' : {
        'lw' : 1,
        'color' : PMF_COLOUR,
        'zorder' : 3,
        'joinstyle' : 'round',
        'capstyle' : 'round',
    },
}


class Distribution(object):

    def E(self):
        return self.mean

    def V(self):
        return self.variance

    def sigma(self):
        return np.sqrt(self.variance)

    def name(self, digits=4):
        name = r'$\mathrm{' + self.__class__.__name__ + '}('
        for key in self.params:
            value = getattr(self, key)
            if isinstance(value, float):
                value = int(value*10**digits)/10**digits
            name += f'{key}={value},'
        name = name[:-1] + ')$'
        return name

class DiscreteDistribution(Distribution):

    def __init__(self, minimum=None, maximum=None):
        self.minimum = minimum
        self.maximum = maximum
        self.is_discrete = True
        self.is_continuous = False

    def update(self, **kwargs):
        for key in self.params:
            if key not in kwargs:
                kwargs[key] = getattr(self, key)
        self.__init__(**kwargs)

    def range(self, bound=None):
        LB = self.minimum
        UB = self.maximum
        if bound is not None:
            bound = int(bound)
            assert bound >= 0
            if LB is None:
                LB = - bound
            if UB is None:
                UB = bound
        assert LB is not None and UB is not None
        assert isinstance(LB, int) and isinstance(UB, int)
        assert LB <= UB
        return np.arange(LB, UB + 1)

    def in_range(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if self.minimum is None:
            LB = True
        else:
            LB = x >= self.minimum
        if self.maximum is None:
            UB = True
        else:
            UB = x <= self.maximum
        if isinstance(x, np.ndarray):
            return (x.astype(int) == x) & LB & UB
        else:
            return x == int(x) and LB and UB

    def pmf(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            y = np.zeros_like(x, dtype=float)
            in_range = self.in_range(x)
            y[in_range] = self.function(x[in_range])
            return y
        else:
            if self.in_range(x):
                return self.function(x)
            else:
                return 0.

    def cdf(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            y = self.cumulative_function(x[0]) + self.pmf(x)
            y = np.cumsum(y)
        else:
            y = self.cumulative_function(x)
        return y

class Binomial(DiscreteDistribution):

    def __init__(self, n=1, p=0.5):
        super().__init__(minimum=0, maximum=n)
        assert isinstance(n, int) and n >= 0
        assert 0 <= p and p <= 1
        self.n = n
        self.p = p
        self.params = ['n', 'p']
        self.mean = self.n*self.p
        self.variance = self.n*self.p*(1 - self.p)

    def function(self, x):
        return self.p**x*(1 - self.p)**(self.n - x)*factorial(self.n)/factorial(x)/factorial(self.n - x)

    def cumulative_function(self, x):
        return 0

class Bernoulli(Binomial):

    def __init__(self, p=0.5):
        super().__init__(minimum=0, maximum=1, n=1, p=p)
        self.params = ['p']

class Geometric(DiscreteDistribution):

    def __init__(self, p=0.5):
        super().__init__(minimum=1)
        assert 0 < p and p <= 1
        self.p = p
        self.params = ['p']
        self.mean = 1/self.p
        self.variance = (1 - self.p)/self.p**2

    def function(self, x):
        return self.p*(1 - self.p)**(x - 1)

    def cumulative_function(self, x):
        return 0


class DistributionPlot(plot):

    def __init__(self):
        super().__init__(**PARAMS)

    def file_name(self):
        return 'distribution'

    def get_ticks(self, bounds, extras, axis, step=1):
        if extras is None:
            extras = self.extra_bottom, self.extra_top
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
            x= - self.label_xshift,
            y= - self.label_yshift,
            height=self.label_height,
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
                    xy=(xt, - self.tick_height/2),
                    width=0,
                    height=self.tick_height,
                    **self.axis_params
                )
                label = self.path_from_string(
                    s=str(xt),
                    x=xt - self.label_xshift,
                    y= - self.label_yshift,
                    height=self.label_height,
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
                    xy=(- tick_width/2, yt),
                    width=tick_width,
                    height=0,
                    **self.axis_params
                )
                label = self.path_from_string(
                    s=str(yt),
                    x= - self.label_xshift,
                    y=yt - self.label_yshift,
                    height=self.label_height,
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

    def plot_discrete(self, distribution):
        x = distribution.range(max(self.xmax, - self.xmin))
        pmf = distribution.pmf(x)
        self.ax.plot(x, pmf, **self.pmf_marker_params)
        for k, y in zip(x, pmf):
            self.plot_shape(
                shape_name='Rectangle',
                xy=(k, 0),
                width=0,
                height=y,
                **self.pmf_bar_params
            )
        x = np.concatenate([[self.xmin - 1], x, [self.xmax + 1]])
        cdf = distribution.cdf(x)
        for k1, k2, y in zip(x[:-1], x[1:], cdf):
            self.plot_shape(
                shape_name='Rectangle',
                xy=(k1, y),
                width=k2 - k1,
                height=0,
                **self.cdf_bar_params
            )
        for k, y1, y2 in zip(x[1:], cdf[:-1], cdf[1:]):
            self.plot_shape(
                shape_name='Rectangle',
                xy=(k, y1),
                width=0,
                height=y2 - y1,
                **self.cdf_bar_params
            )
        self.ax.plot(x[1:], cdf[:-1], **self.cdf_marker_params)
        self.ax.plot(x[1:], cdf[:-1], **self.cdf_empty_marker_params)
        self.ax.plot(x, cdf, **self.cdf_marker_params)

    def plot_fluctuations(self, distribution):
        self.plot_shape(
            shape_name='Rectangle',
            xy=(distribution.E(), self.mean_shift - self.mean_height/2),
            width=0,
            height=self.mean_height,
            **self.fluctuation_params
        )
        self.plot_shape(
            shape_name='Rectangle',
            xy=(distribution.E() - self.std_mult*distribution.sigma(), self.mean_shift),
            width=2*self.std_mult*distribution.sigma(),
            height=0,
            **self.fluctuation_params
        )
        self.ax.plot(
            distribution.E() - self.std_mult*distribution.sigma(),
            self.mean_shift,
            marker='<',
            **self.fluctuation_ends_params
        )
        self.ax.plot(
            distribution.E() + self.std_mult*distribution.sigma(),
            self.mean_shift,
            marker='>',
            **self.fluctuation_ends_params
        )

    def plot_name(self, distribution):
        self.plot_shape(
            shape_name='PathPatch',
            path=self.path_from_string(s=distribution.name(), **self.text_params),
            **self.name_params
        )

    def plot(self, distribution, bound):
        bounds = distribution.range(bound)
        bounds = np.min(bounds), np.max(bounds)
        xticks = self.get_ticks(
            bounds=bounds,
            extras=(self.extra_left, self.extra_right),
            axis='x',
        ).astype(int)
        yticks = self.get_ticks(
            bounds=(0, 1),
            extras=(self.extra_bottom, self.extra_top),
            axis='y',
            step=0.25,
        )
        self.x_over_y = (self.xmax - self.xmin)/(self.ymax - self.ymin)*self.figsize[1]/self.figsize[0]
        self.__figure__()
        self.plot_axis(xticks, yticks)
        self.plot_discrete(distribution)
        self.plot_fluctuations(distribution)
        self.plot_name(distribution)

    def image(self, distribution, bound):
        self.plot(distribution, bound)
        self.save_image(name=self.file_name())

    def video(self, distribution, bound):
        for n, p in enumerate(np.arange(0 , 1, step=0.05)):
            distribution.update(n=n+1, p=p)
            self.plot(distribution, bound)
            self.new_frame()
        self.save_video(name=self.file_name())

    def run(self, distribution, bound=None):
        self.image(distribution, bound)
        self.reset()
        # self.video(distribution, bound)


if __name__ == '__main__':
    DP = DistributionPlot()
    DP.new_param('--bound', type=int, default=None)
    B = Binomial(10)
    DP.run(B)