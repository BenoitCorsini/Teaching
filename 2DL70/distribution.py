import numpy as np
from scipy.special import factorial
from scipy.stats import norm


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
            LB = np.ones_like(x) > 0
        else:
            LB = x >= self.minimum
        if self.maximum is None:
            UB = np.ones_like(x) > 0
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


class ContinuousDistribution(Distribution):

    def __init__(self, minimum=None, maximum=None):
        self.minimum = minimum
        self.maximum = maximum
        self.is_discrete = False

    def update(self, **kwargs):
        for key in self.params:
            if key not in kwargs:
                kwargs[key] = getattr(self, key)
        self.__init__(**kwargs)

    def range(self, bound=None, step=0.1):
        LB = self.minimum
        UB = self.maximum
        if bound is not None:
            assert bound >= 0
            if LB is None:
                LB = - bound
            else:
                UB = bound
            if UB is None:
                UB = bound
        assert LB is not None and UB is not None
        assert LB <= UB
        return np.arange(LB, UB + step, step=step)

    def in_range(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if self.minimum is None:
            LB = np.ones_like(x) > 0
        else:
            LB = x >= self.minimum
        if self.maximum is None:
            UB = np.ones_like(x) > 0
        else:
            UB = x <= self.maximum
        return LB & UB

    def pdf(self, x):
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
        return self.cumulative_function(x)


class ContinuousUniform(ContinuousDistribution):

    def __init__(self, a=0, b=1):
        super().__init__(minimum=a, maximum=b)
        assert a <= b
        self.a = a
        self.b = b
        self.params = ['a', 'b']
        self.mean = (self.a + self.b)/2
        self.variance = ((self.b - self.a)**2)/12

    def function(self, x):
        return 1/(self.b - self.a)

    def cumulative_function(self, x):
        y = (x - self.a)/(self.b - self.a)
        y = y*(y > 0 )*(y < 1) + (y >= 1)
        return y

    @staticmethod
    def params_list(n_steps=53, min_bound=1, max_bound=10, size_return=2):
        ps = [(0, min_bound)]
        for i in range(n_steps):
            ps.append((0, min_bound + (max_bound - min_bound)*(i + 1)/n_steps))
        for i in range(n_steps):
            ps.append(((max_bound - min_bound)*(i + 1)/n_steps, max_bound))
        for i in range(n_steps):
            ratio = (i + 1)/n_steps
            ps.append((
                max(0, (max_bound - min_bound)*(1 - ratio) + (min_bound - size_return)*ratio),
                min(max_bound, (max_bound - min_bound)*(1 - ratio) + (min_bound - size_return)*ratio + size_return)
            ))
        return [{'a' : a, 'b' : b} for (a, b) in ps]

class Exponential(ContinuousDistribution):

    def __init__(self, l=1):
        super().__init__(minimum=0)
        assert l > 0
        self.l = l
        self.params = ['l']
        self.params_name = {'l' : r'\lambda'}
        self.mean = 1/self.l
        self.variance = 1/self.l**2

    def function(self, x):
        return self.l*np.exp(-self.l*x)

    def cumulative_function(self, x):
        y = 1 - np.exp(-self.l*x)
        y = y*(x >= 0)
        return y

    @staticmethod
    def params_list(n_steps=160, min_l=0.01):
        ls = (1 - np.cos(2*np.pi*np.arange(n_steps + 1)/n_steps))/2
        ls = (1 - ls) + ls*min_l
        return [{'l' : l} for l in ls]

class Normal(ContinuousDistribution):

    def __init__(self, mu=0, sigma_square=1):
        super().__init__()
        assert sigma_square > 0
        self.mu = mu
        self.sigma_square = sigma_square
        self.params = ['mu', 'sigma_square']
        self.params_name = {'mu' : r'\mu', 'sigma_square' : r'\sigma^2'}
        self.mean = self.mu
        self.variance = self.sigma_square

    def function(self, x):
        return np.exp(- ((x - self.mu)**2)/2/self.sigma_square)/np.sqrt(2*np.pi*self.sigma_square)

    def cumulative_function(self, x):
        return norm.cdf((x - self.mu)/self.sigma_square**0.5)

    @staticmethod
    def params_list(n_steps=160, mu_shift=8, min_sigma=0.4, max_sigma=10):
        rs = np.sin(2*np.pi*np.arange(n_steps + 1)/n_steps)
        rs[-1] = 0
        mus = mu_shift*rs
        sigma_squares = max_sigma**(rs*(1 + rs))/min_sigma**(rs*(1 - rs))
        return [{'mu' : mu, 'sigma_square' : sigma_square} for (mu, sigma_square) in zip(mus, sigma_squares)]