import numpy as np
import pandas as pd
# Setting up the problem

# S = price of the underlying asset, K = option strike price, r = risk-free rate, T = time to maturity,
# n = number of time steps, u = up movement factor, d = down movement factor

# Underlying price at timestep n: S_n = S_0 * u^n * d^(n-m),
# where m is the number of up movements (n-m is the number of down movements)
# representing the tree in code here using nodes, (i,j) where i is the time step and j is the number of up movements

# Call option value at timestep n: C_n = max(S_n - K, 0)
# Put option value at timestep n: P_n = max(K - S_n, 0)

# initialise the parameters
S_0 = 100
K = 100
r = 0.1194 # annual risk-free rate
T = 0.25 # in years
n = 10
u = 1.03775
d = 1/u
# risk-neutral probability
q = (np.exp(r * T/n) - d) / (u - d)
option_type = 'put'

# initialise the underlying price tree as a matrix
def initialise_underlying_price_tree(S_0, u, d, n):
    price_tree = np.zeros((n+1, n+1))
    price_tree[0,0] = S_0
    for i in range(1, n+1):
        price_tree[i,0] = price_tree[i-1,0] * u
        for j in range(1, i+1):
            price_tree[i,j] = price_tree[i-1,j-1] * d

    price_df = pd.DataFrame(price_tree)
    print(price_df)
    return price_tree

# initialise the option value tree as a matrix
def find_option_value_tree(price_tree, K, r, u, d, n, T, q, option_type):
    option_tree = np.zeros((n+1, n+1))
    # option value at maturity = max(S_n - K, 0) for call option and max(K - S_n, 0) for put option
    payoff_calc_factor = 1 if option_type == 'call' else -1
    option_tree[n] = np.maximum(payoff_calc_factor * (price_tree[n] - K), 0)

    # work backwards through the tree to find the option value at each node
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            option_tree[i,j] = np.exp(-r * T/n) * (q * option_tree[i+1,j] + (1-q) * option_tree[i+1,j+1])

    option_df = pd.DataFrame(option_tree)
    print(option_df)
    return option_tree

def find_american_option_value_tree(price_tree, K, r, u, d, n, T, q, option_type):
    payoff_calc_factor = 1 if option_type == 'call' else -1
    if payoff_calc_factor == 1:
        print("American call option, don't exercise early")
        return
    else:
        american_option_tree = np.zeros((n+1, n+1))
        american_option_tree[n] = np.maximum(K - price_tree[n], 0)
        for i in range(n-1, -1, -1):
            for j in range(i+1):
                american_option_tree[i, j]  = np.maximum(np.maximum(K - price_tree[i, j], 0), np.exp(-r * T/n) * (q * american_option_tree[i+1,j] + (1-q) * american_option_tree[i+1,j+1]))
        american_option_df = pd.DataFrame(american_option_tree)
        print(american_option_df)
        return american_option_tree

def when_should_exercise_american_option_early(american_option_tree, european_option_tree):
    decision_matrix = np.where(american_option_tree > european_option_tree, 1, 0)
    print(pd.DataFrame(decision_matrix))
    stop = False
    i = 0
    while not stop:
        if 1 in decision_matrix[i]:
            print(f"First possible time of early exercise with American option: {i}")
            stop = True
        i += 1


price_tree = initialise_underlying_price_tree(S_0, u, d, n)
option_value_tree = find_option_value_tree(price_tree, K, r, u, d, n, T, q, option_type)
american_option_tree = find_american_option_value_tree(price_tree, K, r, u, d, n, T, q, option_type)
decision_tree = when_should_exercise_american_option_early(american_option_tree, option_value_tree)
