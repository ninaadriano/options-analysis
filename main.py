from binomial_pricing_model import BinomialPricingModel
from black_scholes import BlackScholesModel
from stock_price_simulator import GBMStockPriceSimulator

S_0=100
K=100
T=0.25
n=10
u=1.03775
r=0.1194
sigma=0.2
gbm_steps=252
num_paths=100
option_type='put'

binomial_model = BinomialPricingModel(S_0=S_0, K=K, r=r, T=T, n=n, u=u, option_type=option_type)
black_scholes_model = BlackScholesModel(S_0=S_0, K=K, T=T, r=r, sigma=sigma)
simulated_stock_prices = GBMStockPriceSimulator(S_0=S_0, T=T, r=r, sigma=sigma, gbm_steps=gbm_steps, num_paths=num_paths)

print(f"Binomial Model Option Price: {binomial_model.option_price()}")
print(f"Black-Scholes Model Option Price: {black_scholes_model.option_price(option_type)}")
print(f"Simulated GBM Model Option Price: {simulated_stock_prices.calculate_gbm_price(K, option_type)}")