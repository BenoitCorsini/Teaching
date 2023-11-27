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
    # 'dpi' : 10,
    'extra_left' : 0.2,
    'extra_right' : 0.2,
    'extra_bottom' : 0.1,
    'extra_top' : 0.15,
    'max_ticks' : 15,
    'tick_height' : 0.01,
    'label_xshift' : 0.1,
    'label_yshift' : 0.01,
    'label_height' : 0.02,
    'pmf_bar' : 0.15,
    'pmf_dot' : 0.3,
    'cdf_bar' : 0.1,
    'cdf_dot' : 0.2,
    'cdf_inn' : 0.08,
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
    'pmf_params' : {
        'lw' : 0,
        'color' : PMF_COLOUR,
        'zorder' : 1,
    },
    'cdf_params' : {
        'lw' : 0,
        'color' : CDF_COLOUR,
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
    'times' : {
        'initial' : 1,
        'final' : 1,
        'binomial_and_poisson' : 8,
    },
}


class Distribution(object):

    def E(self):
        return self.mean

    def V(self):
        return self.variance

    def sigma(self):
        return np.sqrt(self.variance)

    def name(self, digits=2):
        name = r'$\mathrm{' + self.__class__.__name__ + '}('
        for key in self.params:
            value = getattr(self, key)
            if hasattr(self, 'params_name'):
                key = self.params_name.get(key, key)
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
            else:
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

class DiscreteUniform(DiscreteDistribution):

    def __init__(self, a=0, b=1):
        super().__init__(minimum=a, maximum=b)
        assert isinstance(a, int)
        assert isinstance(b, int)
        assert a <= b
        self.a = a
        self.b = b
        self.params = ['a', 'b']
        self.mean = (self.a + self.b)/2
        self.variance = ((self.b - self.a + 1)**2 - 1)/12

    def function(self, x):
        return 1/(self.b - self.a + 1)

    def cumulative_function(self, x):
        return 0

    @staticmethod
    def params_list(n_steps=5, max_bound=10, size_return=2):
        ps = [(0, 0)]*int(n_steps/2)
        for i in range(max_bound):
            ps += [(0, i + 1)]*n_steps
        for i in range(max_bound):
            ps += [(i + 1, max_bound)]*n_steps
        for i in range(size_return):
            ps += [(max_bound - i - 1, max_bound)]*n_steps
        for i in range(max_bound - 1):
            ps += [(max(0, max_bound - size_return - i - 1), max_bound - i - 1)]*n_steps
        ps += [(0, 0)]*(n_steps - int(n_steps/2))
        return [{'a' : int(a), 'b' : int(b)} for (a, b) in ps]

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

    @staticmethod
    def params_list(n_steps=160):
        ps = (1 - np.cos(2*np.pi*np.arange(n_steps + 1)/n_steps))/2
        return [{'p' : p} for p in ps]

class Bernoulli(Binomial):

    def __init__(self, p=0.5):
        super().__init__(n=1, p=p)
        self.params = ['p']

    @staticmethod
    def params_list(n_steps=160):
        ps = (1 - np.cos(2*np.pi*np.arange(n_steps + 1)/n_steps))/2
        return [{'p' : p} for p in ps]

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

    @staticmethod
    def params_list(n_steps=160, min_p=0.01):
        ps = (1 - np.cos(2*np.pi*np.arange(n_steps + 1)/n_steps))/2
        ps = (1 - ps) + ps*min_p
        return [{'p' : p} for p in ps]

class Poisson(DiscreteDistribution):

    def __init__(self, l=1):
        super().__init__(minimum=0)
        assert l >= 0
        self.l = l
        self.params = ['l']
        self.params_name = {'l' : r'\lambda'}
        self.mean = self.l
        self.variance = self.l

    def function(self, x):
        return self.l**x*np.exp(-self.l)/factorial(x)

    def cumulative_function(self, x):
        return 0

    @staticmethod
    def params_list(n_steps=160, max_l=16):
        ls = (1 - np.cos(2*np.pi*np.arange(n_steps + 1)/n_steps))/2
        return [{'l' : max_l*l} for l in ls]


class DistributionPlot(plot):

    def __init__(self):
        super().__init__(**PARAMS)

    def file_name(self, distribution):
        return distribution.__class__.__name__

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

    def plot_discrete(self, distribution, pmf_params={}, cdf_params={}):
        self.plot_pmf(distribution, **pmf_params)
        self.plot_cdf(distribution, **cdf_params)

    def plot_pmf(self, distribution, extra_key='', **kwargs):
        for k, infos in self.pmf.items():
            infos['bar'].set_visible(False)
            infos['dot'].set_visible(False)
        for key, value in self.pmf_params.items():
            kwargs[key] = kwargs.get(key, value)
            kwargs['visible'] = True
        x = distribution.range(max(self.xmax, - self.xmin))
        x = x[distribution.in_range(x)]
        pmf = distribution.pmf(x)
        for k, y in zip(x, pmf):
            key = f'{extra_key}{k}'
            if key not in self.pmf:
                self.pmf[key] = {
                    'bar' : self.plot_shape(
                        shape_name='Rectangle',
                        xy=(k - self.pmf_bar/2, 0),
                        width=self.pmf_bar,
                        height=0,
                    ),
                    'dot' : self.plot_shape(
                        shape_name='Ellipse',
                        xy=(0, 0),
                        width=self.pmf_dot,
                        height=self.pmf_dot/self.x_over_y,
                    ),
                }
            self.pmf[key]['bar'].set_height(y)
            self.pmf[key]['bar'].set(**kwargs)
            self.pmf[key]['dot'].set_center((k, y))
            self.pmf[key]['dot'].set(**kwargs)

    def plot_cdf(self, distribution, extra_key='', **kwargs):
        for infos in self.cdf.values():
            for p in infos:
                p.set_visible(False)
        for key, value in self.cdf_params.items():
            kwargs[key] = kwargs.get(key, value)
        x = distribution.range(max(self.xmax, - self.xmin))
        x = x[distribution.in_range(x)]
        x = np.concatenate([[self.xmin - 1], x, [self.xmax + 1]])
        cdf = distribution.cdf(x)
        infos = []
        for k1, k2, y in zip(x[:-1], x[1:], cdf):
            infos.append(self.plot_shape(
                shape_name='Rectangle',
                xy=(k1, y - self.cdf_bar/self.x_over_y/2),
                width=k2 - k1,
                height=self.cdf_bar/self.x_over_y,
                **kwargs
            ))
        for k, y1, y2 in zip(x[1:], cdf[:-1], cdf[1:]):
            infos.append(self.plot_shape(
                shape_name='Rectangle',
                xy=(k - self.cdf_bar/2, y1),
                width=self.cdf_bar,
                height=y2 - y1,
                **kwargs
            ))
        for k, y in zip(x[1:], cdf[:-1]):
            infos.append(self.plot_shape(
                shape_name='Ellipse',
                xy=(k, y),
                width=self.cdf_dot,
                height=self.cdf_dot/self.x_over_y,
                **kwargs
            ))
        for k, y in zip(x[1:], cdf[:-1]):
            shape = self.plot_shape(
                shape_name='Ellipse',
                xy=(k, y),
                width=self.cdf_inn,
                height=self.cdf_inn/self.x_over_y,
                **kwargs
            )
            shape.set_color('white')
            infos.append(shape)
        for k, y in zip(x, cdf):
            infos.append(self.plot_shape(
                shape_name='Ellipse',
                xy=(k, y),
                width=self.cdf_dot,
                height=self.cdf_dot/self.x_over_y,
                **kwargs
            ))
        self.cdf[key] = infos

    def plot_fluctuations(self, distribution):
        self.mean.set_x(distribution.E())
        self.std.set_x(distribution.E() - self.std_mult*distribution.sigma())
        self.std.set_width(2*self.std_mult*distribution.sigma())

    def plot_name(self, distribution):
        self.name.set_path(self.path_from_string(s=distribution.name(), **self.text_params))

    def setup(self, distribution, bound):
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
        self.reset()
        self.plot_axis(xticks, yticks)
        self.pmf = {}
        self.cdf = {}
        self.mean = self.plot_shape(
            shape_name='Rectangle',
            xy=(0, self.mean_shift - self.mean_height/2),
            width=0,
            height=self.mean_height,
            **self.fluctuation_params
        )
        self.std = self.plot_shape(
            shape_name='Rectangle',
            xy=(0, self.mean_shift),
            width=0,
            height=0,
            **self.fluctuation_params
        )
        self.name = self.plot_shape(
            shape_name='PathPatch',
            path=self.path_from_string(s='.', **self.text_params),
            **self.name_params
        )

    def image(self, distribution, bound):
        self.setup(distribution, bound)
        self.update_discrete(distribution)
        self.save_image(name=self.file_name(distribution))

    def update_discrete(self, distribution):
        self.plot_discrete(distribution)
        self.plot_fluctuations(distribution)
        self.plot_name(distribution)

    def evolution(self, distribution, bound):
        self.reset()
        params_list = distribution.params_list()
        distribution.update(**params_list[0])
        self.setup(distribution, bound)
        self.update_discrete(distribution)
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        for params in params_list:
            distribution.update(**params)
            self.update_discrete(distribution)
            self.new_frame()
        for _ in range(int(self.fps*self.times['final'])):
            self.new_frame()
        self.save_video(name=self.file_name(distribution))

    def run(self, distribution, bound=None, **kwargs):
        self.image(distribution, bound)
        self.evolution(distribution, bound)

    def binomial_and_poisson(self, bound=None, n_max=10, l=1):
        self.reset()
        B = Binomial()
        P = Poisson(l=l)
        self.plot(P, bound=bound)
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame() 
        self.save_video(name=f'BinomialPoisson(lambda={l})')       


if __name__ == '__main__':
    DP = DistributionPlot()
    DP.new_param('--bound', type=int, default=1)
    DP.new_param('--n_max', type=int, default=1)
    DP.new_param('--l', type=float, default=1)
    X = Poisson()
    DP.run(X, **DP.get_kwargs())
    # DP.binomial_and_poisson(**DP.get_kwargs())