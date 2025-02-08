import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    # Using analogous parameters to previously-implemented binomial model
    # Note q is the dividend yield, currently defaulting to 0 (ignoring dividends)
    def __init__(self, S_0, K, T, r, sigma, q=0):
        self.S_0 = S_0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    # Black-Scholes pricing formulae for European call and put options
    # Call option price: C = S_0 * e^(-qT) * N(d_1) - K * e^(-rT) * N(d_2)
    # Put option price: P = K * e^(-rT) * N(-d_2) - S_0 * e^(-qT) * N(-d_1)
    # Where N is the CDF of the standard normal distribution and
    # d_1 = (ln(S_0 / K) + (r - q + sigma^2 / 2)T) / (sigma * sqrt(T))
    # d_2 = d_1 - sigma * sqrt(T)

    def d1(self):
        return (np.log(self.S_0 / self.K) + (self.r - self.q + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def _call_price(self):
        return self.S_0 * np.exp(-self.q * self.T) * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())

    def _put_price(self):
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S_0 * np.exp(-self.q * self.T) * norm.cdf(-self.d1())

    def option_price(self, _option_type = 'call'):
        if _option_type == 'call':
            return self._call_price()
        elif _option_type == 'put':
            return self._put_price()
        elif _option_type == 'both':
            return {'call': self._call_price(), 'put': self._put_price()}
        else:
            raise ValueError("Option type must be 'call', 'put', or 'both'")

model=BlackScholesModel(S_0=100, K=100, T=0.25, r=0.1194, sigma=0.2)
print(model.option_price('put'))