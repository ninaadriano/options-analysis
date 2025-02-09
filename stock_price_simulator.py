import numpy as np
import matplotlib.pyplot as plt

class GBMStockPriceSimulator:
    def __init__(self, S_0, T, r, sigma, gbm_steps, num_paths, c=0):
        # S_0 = Stock Price at time 0
        # K = Strike Price
        # T = Time to maturity in years
        # r = risk-free interest rate
        # c = dividend yield
        # sigma = volatility
        self.S_0 = S_0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.gbm_steps = gbm_steps
        self.num_paths = num_paths
        self.c = c

    def gbm_paths(self):
        dt = self.T / self.gbm_steps
        log_ST = np.log(self.S_0) + np.cumsum(((self.r - self.c - self.sigma ** 2 / 2) * dt + self.sigma * np.sqrt(dt) * np.random.normal(size=(self.gbm_steps, self.num_paths))), axis=0)
        # [gbm_steps, num_paths] Matrix of asset paths
        return np.exp(log_ST)

    def plot_gbm_paths(self):
        paths = self.gbm_paths()
        plt.plot(paths)
        plt.xlabel('Time Steps')
        plt.ylabel('Stock Price')
        plt.title('GBM Paths')
        plt.show()

    def calculate_gbm_price(self, K, option_type):
        # 1. find the paths simulated by the GBM for stock price
        paths = self.gbm_paths()
        # 2. find the payoff of the option at maturity for each path
        payoffs = np.maximum(paths[-1] - K, 0) if option_type == 'call' else np.maximum(K - paths[-1], 0)
        # 3. find the expected price of the option based on the payoffs discounted back to time 0
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return price

# Example usage
simulator = GBMStockPriceSimulator(S_0=100, T=0.25, r=0.1194, sigma=0.2, gbm_steps=252, num_paths=10)
simulator.plot_gbm_paths()
option_price = simulator.calculate_gbm_price(option_type='put', K=100)
print(f"Option Price: {option_price}")