import sys
import numpy as np

sys.path.append('../')
from bcplot import BCPlot as plot
from distribution import *

CMAP = plot.get_cmap([
    (255/255, 255/255, 255/255),
    ( 64/255, 149/255, 191/255),
    (  0/255,   0/255,   0/255),
])
PMDF_COLOUR = CMAP(0.7)
CDF_COLOUR = CMAP(0.3)
FLU_COLOUR = CMAP(0.5)
PARAMS = {
    # 'dpi' : 10,
    'extra_left' : 0.1,
    'extra_right' : 0.1,
    'extra_bottom' : 0.1,
    'extra_top' : 0.1,
    'max_ticks' : 15,
    'tick_height' : 0.01,
    'label_xshift' : 0.1,
    'label_yshift' : 0.01,
    'label_height' : 0.02,
    'pmf_bar' : 0.15,
    'pmf_dot' : 0.3,
    'cdf_bar' : 0.15,
    'cdf_dot' : 0.15,
    'cdf_inn' : 0.05,

    'label_xshift' : 0.04,
    'cdf_bar' : 0.1,
    'cdf_dot' : 0.1,
    'cdf_inn' : 0.05,

    'mean_shift' : - 0.05,
    'mean_height' : 0.015,
    'std_mult' : 1.96,
    'arrow_height' : 0.015,
    'arrow_width' : 0.005,
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
        'color' : PMDF_COLOUR,
        'zorder' : 1,
    },
    'pdf_params' : {
        'lw' : 8,
        'color' : PMDF_COLOUR,
        'zorder' : 1,
        'joinstyle' : 'round',
        'capstyle' : 'round',
        'fill' : False,
        'closed' : False,
    },
    'discrete_cdf_params' : {
        'lw' : 0,
        'color' : CDF_COLOUR,
        'zorder' : 0,
    },
    'continuous_cdf_params' : {
        'lw' : 8,
        'color' : CDF_COLOUR,
        'zorder' : 0,
        'joinstyle' : 'round',
        'capstyle' : 'round',
        'fill' : False,
        'closed' : False,
    },
    'fluctuation_params' : {
        'color' : FLU_COLOUR,
        'lw' : 3,
        'zorder' : 0,
        'capstyle' : 'round',
        'joinstyle' : 'round',
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
    'text_params' : {
        'x' : 0 + 0.15,
        'y' : 1 + 0.01,
        'anchor' : 'south west',
        'height' : 0.03,
    },
    'name_params' : {
        'lw' : 1,
        'color' : PMDF_COLOUR,
        'zorder' : 3,
        'joinstyle' : 'round',
        'capstyle' : 'round',
    },
    'times' : {
        'initial' : 1,
        'final' : 1,
        'binomial_and_poisson' : 8,
        'student_and_normal' : 8,
        'discrete_and_continuous' : 8,
    },
}


class DistributionPlot(plot):

    def __init__(self):
        super().__init__(**PARAMS)

    def file_name(self, distribution):
        return distribution.__class__.__name__

    def setup(self, distribution, bound):
        bounds = distribution.range(bound)
        bounds = np.min(bounds), np.max(bounds)
        xticks = self.get_ticks(
            bounds=bounds,
            extras=(self.extra_left, self.extra_right),
            axis='x',
        ).astype(int)
        yticks = np.round(self.get_ticks(
            bounds=(0, 1),
            extras=(self.extra_bottom, self.extra_top),
            axis='y',
            step=0.25,
        ), 2)
        self.x_over_y = (self.xmax - self.xmin)/(self.ymax - self.ymin)*self.figsize[1]/self.figsize[0]
        self.reset()
        self.plot_axis(xticks, yticks)
        self.pmf = {}
        self.pdf = {}
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
        self.arrows = {}
        for m in [- 1, 1]:
            self.arrows[m] = self.plot_shape(
                shape_name='Polygon',
                xy=[[0, 0]],
                fill=False,
                closed=False,
                **self.fluctuation_params
            )
        self.name = self.plot_shape(
            shape_name='PathPatch',
            path=self.path_from_string(s='.'),
            **self.name_params
        )

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

    def plot_fluctuations(self, distribution, **kwargs):
        self.mean.set_x(distribution.E())
        self.mean.set(**kwargs)
        self.std.set_x(distribution.E() - self.std_mult*distribution.sigma())
        self.std.set_width(2*self.std_mult*distribution.sigma())
        self.std.set(**kwargs)
        for m in [- 1, 1]:
            xy = np.array([
                [
                    distribution.E() + m*self.std_mult*distribution.sigma() - m*self.arrow_width*self.x_over_y,
                    distribution.E() + m*self.std_mult*distribution.sigma(),
                    distribution.E() + m*self.std_mult*distribution.sigma() - m*self.arrow_width*self.x_over_y,
                ],
                [
                    self.mean_shift + self.arrow_height/2,
                    self.mean_shift,
                    self.mean_shift - self.arrow_height/2,
                ],
            ])
            self.arrows[m].set_xy(xy.T)
            self.arrows[m].set(**kwargs)

    def plot_name(self, distribution, **kwargs):
        self.name.set_path(self.path_from_string(s=distribution.name(), **self.text_params))
        self.name.set(**kwargs)

    def plot_discrete(self, distribution, key='', set_visible=False, pmf_params={}, discrete_cdf_params={}):
        self.plot_pmf(distribution, key=key, set_visible=set_visible, **pmf_params)
        self.plot_discrete_cdf(distribution, key=key, **discrete_cdf_params)

    def plot_pmf(self, distribution, key='', set_visible=False, **kwargs):
        for infos in self.pmf.values():
            infos['bar'].set_visible(set_visible)
            infos['dot'].set_visible(set_visible)
        for k, v in self.pmf_params.items():
            kwargs[k] = kwargs.get(k, v)
            if 'visible' not in kwargs:
                kwargs['visible'] = True
        x = distribution.range(max(self.xmax, - self.xmin))
        x = x[distribution.in_range(x)]
        # x = x[(x >= self.xmin) & (x <= self.xmax)]
        pmf = distribution.pmf(x)
        for k, y in zip(x, pmf):
            new_key = f'{key}{k}'
            if new_key not in self.pmf:
                self.pmf[new_key] = {
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
            self.pmf[new_key]['bar'].set_height(y)
            self.pmf[new_key]['bar'].set(**kwargs)
            self.pmf[new_key]['dot'].set_center((k, y))
            self.pmf[new_key]['dot'].set(**kwargs)

    def plot_discrete_cdf(self, distribution, key='', **kwargs):
        for infos in self.cdf.values():
            for p in infos:
                p.set_visible(False)
        for k, v in self.discrete_cdf_params.items():
            kwargs[k] = kwargs.get(k, v)
        x = distribution.range(max(self.xmax, - self.xmin))
        x = x[distribution.in_range(x)]
        # x = x[(x >= self.xmin) & (x <= self.xmax)]
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

    def update_discrete(self, distribution, key='', set_visible=False, pmf_params={}, discrete_cdf_params={}, fluctuation_params={}, name_params={}):
        self.plot_discrete(distribution, key=key, set_visible=set_visible, pmf_params=pmf_params, discrete_cdf_params=discrete_cdf_params)
        self.plot_fluctuations(distribution, **fluctuation_params)
        self.plot_name(distribution, **name_params)

    def plot_continuous(self, distribution, key='', set_visible=False, pmf_params={}, discrete_cdf_params={}):
        self.plot_pdf(distribution, key=key, set_visible=set_visible, **pmf_params)
        self.plot_continuous_cdf(distribution, key=key, set_visible=set_visible, **discrete_cdf_params)

    def plot_pdf(self, distribution, key='', set_visible=False, **kwargs):
        for infos in self.pdf.values():
            infos.set_visible(set_visible)
        for k, v in self.pdf_params.items():
            kwargs[k] = kwargs.get(k, v)
            if 'visible' not in kwargs:
                kwargs['visible'] = True
        if key not in self.pdf:
            self.pdf[key] = self.plot_shape(
                shape_name='Polygon',
                xy=[[0,0]],
            )
        x = distribution.range(max(self.xmax, - self.xmin))
        x = x[distribution.in_range(x)]
        pdf = distribution.pdf(x)
        # x = np.concatenate([[self.xmin,x[0]],x,[x[-1],self.xmax]])
        # pdf = np.concatenate([[0,0],pdf,[0,0]])
        self.pdf[key].set_xy(np.stack([x, pdf], axis=-1))
        self.pdf[key].set(**kwargs)

    def plot_continuous_cdf(self, distribution, key='', set_visible=False, **kwargs):
        for infos in self.cdf.values():
            for p in infos:
                p.set_visible(False)
        for k, v in self.continuous_cdf_params.items():
            kwargs[k] = kwargs.get(k, v)
            if 'visible' not in kwargs:
                kwargs['visible'] = True
        if key not in self.cdf:
            self.cdf[key] = [self.plot_shape(
                shape_name='Polygon',
                xy=[[0,0]],
            )]
        x = distribution.range(max(self.xmax, - self.xmin))
        x = x[distribution.in_range(x)]
        cdf = distribution.cdf(x)
        x = np.concatenate([[self.xmin,x[0]],x,[x[-1],self.xmax]])
        cdf = np.concatenate([[0,0],cdf,[1,1]])
        self.cdf[key][0].set_xy(np.stack([x, cdf], axis=-1))
        self.cdf[key][0].set(**kwargs)

    def update_continuous(self, distribution, key='', set_visible=False, pmf_params={}, discrete_cdf_params={}, fluctuation_params={}, name_params={}):
        self.plot_continuous(distribution, key=key, set_visible=set_visible, pmf_params=pmf_params, discrete_cdf_params=discrete_cdf_params)
        self.plot_fluctuations(distribution, **fluctuation_params)
        self.plot_name(distribution, **name_params)

    def image(self, distribution, bound):
        self.setup(distribution, bound)
        self.update(distribution)
        self.save_image(name=self.file_name(distribution))

    def update(self, distribution, **kwargs):
        if distribution.is_discrete:
            self.update_discrete(distribution, **kwargs)
        else:
            self.update_continuous(distribution, **kwargs)

    def evolution(self, distribution, bound):
        self.reset()
        params_list = distribution.params_list()
        distribution.update(**params_list[0])
        self.setup(distribution, bound)
        self.update(distribution)
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        for params in params_list:
            distribution.update(**params)
            self.update(distribution)
            self.new_frame()
        for _ in range(int(self.fps*self.times['final'])):
            self.new_frame()
        self.save_video(name=self.file_name(distribution))

    def run(self, distribution, bound=None, **kwargs):
        self.image(distribution, bound)
        self.evolution(distribution, bound)

    def binomial_and_poisson(self, use_pmf=True, bound=None, n_max=1, l=1, **kwargs):
        self.reset()
        P = Poisson(l=l)
        self.setup(P, bound=bound)
        poisson_text_params = self.text_params.copy()
        poisson_text_params['anchor'] = 'north west'
        poisson_text_params['y'] = 2 - poisson_text_params['y']
        poisson_name_params = self.name_params.copy()
        poisson_name_params['color'] = CDF_COLOUR
        poisson_name_params['zorder'] = self.pmf_params['zorder']
        poisson_name = self.plot_shape(
            shape_name='PathPatch',
            path=self.path_from_string(s=P.name(), **poisson_text_params),
            **poisson_name_params
        )
        poisson_dist_params = self.discrete_cdf_params.copy()
        poisson_dist_params['zorder'] = self.pmf_params['zorder']
        if use_pmf:
            self.plot_pmf(P, key='Poisson', **poisson_dist_params)
        else:
            self.plot_discrete_cdf(P, key='Poisson', **poisson_dist_params)

        B = Binomial()
        params_list = []
        for n in range(1, n_max + 1):
            params_list.append({
                'n' : n,
                'p' : min(l/n, 1),
            })
        time_per_step = int(np.ceil(self.fps*self.times['binomial_and_poisson']/len(params_list)))
        B.update(**params_list[0])
        if use_pmf:
            default_params = {
                'set_visible' : True,
                'discrete_cdf_params' : {'visible' : False},
                'fluctuation_params' : {'visible' : False},
            }
        else:
            default_params = {
                'pmf_params' : {'visible' : False},
                'discrete_cdf_params' : self.pmf_params.copy(),
                'fluctuation_params' : {'visible' : False},
            }
        self.update_discrete(B, **default_params)
        if not use_pmf:
            for p in self.cdf['Poisson']:
                p.set_visible(True)

        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        for params in params_list:
            B.update(**params)
            self.update_discrete(B, **default_params)
            if not use_pmf:
                for p in self.cdf['Poisson']:
                    p.set_visible(True)
            for _ in range(time_per_step):
                self.new_frame()
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        name = 'BinomialPoisson'
        if use_pmf:
            name += '(PMF)'
        else:
            name += '(CDF)'
        self.save_video(name=name)

    def student_and_normal(self, bound=None, nu_max=40, nu_min=2, **kwargs):
        name = 'StudentNormal'
        self.reset()
        N = Normal(mu=0, sigma_square=1)
        self.setup(N, bound=bound)
        normal_text_params = self.text_params.copy()
        normal_text_params['anchor'] = 'north west'
        normal_text_params['y'] = 2 - normal_text_params['y']
        normal_name_params = self.name_params.copy()
        normal_name_params['color'] = CDF_COLOUR
        normal_name_params['zorder'] = self.pdf_params['zorder']
        normal_name = self.plot_shape(
            shape_name='PathPatch',
            path=self.path_from_string(s=r'N(0,1)', **normal_text_params),
            **normal_name_params
        )
        normal_dist_params = self.continuous_cdf_params.copy()
        normal_dist_params['zorder'] = self.pdf_params['zorder']
        self.plot_pdf(N, key='Normal', **normal_dist_params)
        S = Student(nu_min)
        steps = int(self.fps*self.times['student_and_normal'])
        steps = (1 + np.arange(steps))/steps
        steps = steps**3
        params_list = []
        for step in steps:
            params_list.append({'nu' : nu_min + (nu_max - nu_min)*step})
        default_params = {
            'set_visible' : True,
            'discrete_cdf_params' : {'visible' : False},
            'fluctuation_params' : {'visible' : False},
        }
        self.update_continuous(S, **default_params)
        self.save_image(name=name)
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        for params in params_list:
            S.update(**params)
            self.update_continuous(S, **default_params)
            self.new_frame()
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        self.save_video(name=name)

    def discrete_and_continuous(self, scaling_distribution, continuous_distribution, bound=None, max_scale=10, **kwargs):
        self.reset()
        self.setup(continuous_distribution, bound=bound)
        continuous_text_params = self.text_params.copy()
        continuous_text_params['anchor'] = 'north west'
        continuous_text_params['y'] = 2 - continuous_text_params['y']
        continuous_name_params = self.name_params.copy()
        continuous_name_params['color'] = CDF_COLOUR
        continuous_name_params['zorder'] = self.pdf_params['zorder']
        continuous_name = self.plot_shape(
            shape_name='PathPatch',
            path=self.path_from_string(s=continuous_distribution.name(), **continuous_text_params),
            **continuous_name_params
        )
        continuous_dist_params = self.continuous_cdf_params.copy()
        continuous_dist_params['zorder'] = self.pdf_params['zorder']
        self.plot_continuous_cdf(continuous_distribution, key='continuous', **continuous_dist_params)
        params_list = []
        for scale in range(1, max_scale + 1):
            params_list.append({'scale' : scale})
        time_per_step = int(np.ceil(self.fps*self.times['discrete_and_continuous']/len(params_list)))
        scaling_distribution.update(**params_list[0])
        default_params = {
            'pmf_params' : {'visible' : False},
            'discrete_cdf_params' : self.pmf_params.copy(),
            'fluctuation_params' : {'visible' : False},
        }
        self.update(scaling_distribution, **default_params)
        self.cdf['continuous'][0].set_visible(True)
        # self.save_image()
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        for params in params_list:
            scaling_distribution.update(**params)
            self.update(scaling_distribution, **default_params)
            self.cdf['continuous'][0].set_visible(True)
            for _ in range(time_per_step):
                self.new_frame()
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        self.save_video(name=scaling_distribution.__class__.__name__)


if __name__ == '__main__':
    DP = DistributionPlot()
    DP.new_param('--bound', type=int, default=1)
    DP.new_param('--n_max', type=int, default=1)
    DP.new_param('--nu_min', type=float, default=2)
    DP.new_param('--nu_max', type=float, default=40)
    DP.new_param('--l', type=float, default=1)
    DP.new_param('--use_pmf', type=int, default=1)
    DP.new_param('--max_scale', type=int, default=1)
    # X = Poisson()
    # X = Normal()
    # DP.run(X, **DP.get_kwargs())
    # DP.binomial_and_poisson(**DP.get_kwargs())
    DP.student_and_normal(**DP.get_kwargs())
    # scaling_distribution, continuous_distribution = ScalingUniform(), ContinuousUniform()
    # scaling_distribution, continuous_distribution = ScalingGeometric(), Exponential()
    # scaling_distribution, continuous_distribution = ScalingBinomial(), Normal()
    # scaling_distribution, continuous_distribution = ScalingPoisson(), Normal()
    # DP.discrete_and_continuous(
    #     scaling_distribution,
    #     continuous_distribution,
    #     **DP.get_kwargs()
    # )