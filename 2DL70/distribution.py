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
PARAMS = {
}


class Distribution(object):

    def E(self):
        return self.mean

    def V(self):
        return self.variance

    def sigma(self):
        return np.sqrt(self.variance)

    def PMF(self, x):
        assert self.is_discrete
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            y = np.zeros_like(x, dtype=float)
            in_range = self.in_range(x)
            y[in_range] = self.pmf(x[in_range])
            return y
        else:
            if self.in_range(x):
                return self.pmf(x)
            else:
                return 0.

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
            if isinstance(bound, int):
                LB = self.minimum
                if self.maximum is None:
                    UB = bound
            else:
                if self.minimum is None:
                    LB = int(bound[0])
                if self.maximum is None:
                    UB = int(bound[1])
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

    def pmf(self, x):
        return self.p**x*(1 - self.p)**(self.n - x)*factorial(self.n)/factorial(x)/factorial(self.n - x)

class Geometric(DiscreteDistribution):

    def __init__(self, p=0.5):
        super().__init__(minimum=1)
        assert 0 < p and p <= 1
        self.p = p
        self.params = ['p']
        self.mean = 1/self.p
        self.variance = (1 - self.p)/self.p**2

    def pmf(self, x):
        return self.p*(1 - self.p)**(x - 1)


class DistributionPlot(plot):

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
    # DP = DistributionPlot()
    # DP.new_param('--seed', type=int, default=None)
    # DP.run()
    B = Binomial(10)