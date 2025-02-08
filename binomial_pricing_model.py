import numpy as np
import pandas as pd

class BinomialPricingModel:
    # Problem setup for Binomial option pricing model for European and American options
    # S = price of the underlying asset, K = option strike price, r = risk-free rate, T = time to maturity,
    # n = number of time steps, u = up movement factor, d = down movement factor, currently ignoring dividends

    # Underlying price at timestep n: S_n = S_0 * u^n * d^(n-m),
    # where m is the number of up movements (n-m is the number of down movements)
    # representing the tree in code here using nodes, (i,j) where i is the time step and j is the number of up movements

    # Call option value at timestep n: C_n = max(S_n - K, 0)
    # Put option value at timestep n: P_n = max(K - S_n, 0)
    def __init__(self, S_0, K, r, T, n, u, option_type):
        self.S_0 = S_0
        self.K = K
        self.r = r
        self.T = T
        self.n = n
        self.u = u
        self.d = 1 / u
        self.q = (np.exp(r * T / n) - self.d) / (self.u - self.d)
        self.option_type = option_type
        self.price_tree = self.initialise_underlying_price_tree()
        self.option_value_tree = self.find_option_value_tree()
        self.american_option_tree = self.find_american_option_value_tree()

    def initialise_underlying_price_tree(self):
        price_tree = np.zeros((self.n + 1, self.n + 1))
        price_tree[0, 0] = self.S_0
        for i in range(1, self.n + 1):
            price_tree[i, 0] = price_tree[i - 1, 0] * self.u
            for j in range(1, i + 1):
                price_tree[i, j] = price_tree[i - 1, j - 1] * self.d

        price_df = pd.DataFrame(price_tree)
        print(price_df)
        return price_tree

    def find_option_value_tree(self):
        option_tree = np.zeros((self.n + 1, self.n + 1))
        payoff_calc_factor = 1 if self.option_type == 'call' else -1
        option_tree[self.n] = np.maximum(payoff_calc_factor * (self.price_tree[self.n] - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                option_tree[i, j] = np.exp(-self.r * self.T / self.n) * (self.q * option_tree[i + 1, j] + (1 - self.q) * option_tree[i + 1, j + 1])

        option_df = pd.DataFrame(option_tree)
        print(option_df)
        return option_tree

    def find_american_option_value_tree(self):
        payoff_calc_factor = 1 if self.option_type == 'call' else -1
        if payoff_calc_factor == 1:
            print("American call option, don't exercise early")
            return None
        else:
            american_option_tree = np.zeros((self.n + 1, self.n + 1))
            american_option_tree[self.n] = np.maximum(self.K - self.price_tree[self.n], 0)
            for i in range(self.n - 1, -1, -1):
                for j in range(i + 1):
                    american_option_tree[i, j] = np.maximum(np.maximum(self.K - self.price_tree[i, j], 0), np.exp(-self.r * self.T / self.n) * (self.q * american_option_tree[i + 1, j] + (1 - self.q) * american_option_tree[i + 1, j + 1]))
            american_option_df = pd.DataFrame(american_option_tree)
            print(american_option_df)
            return american_option_tree

    def when_should_exercise_american_option_early(self):
        if self.american_option_tree is None:
            return
        decision_matrix = np.where(self.american_option_tree > self.option_value_tree, 1, 0)
        print(pd.DataFrame(decision_matrix))
        stop = False
        i = 0
        while not stop:
            if 1 in decision_matrix[i]:
                print(f"First possible time of early exercise with American option: {i}")
                stop = True
            i += 1

# Example usage
model = BinomialPricingModel(S_0=100, K=100, r=0.1194, T=0.25, n=10, u=1.03775, option_type='put')
model.initialise_underlying_price_tree()
model.find_option_value_tree()
model.find_american_option_value_tree()
model.when_should_exercise_american_option_early()
