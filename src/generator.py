# The Gaussian "True Price" Generator

import numpy as np

'''
This generates V_t, the unobservable Fundamental Value or Intrinsic Value of the asset.
This is the present value of all future cash flows for this asset.
This is distinct from an easily-calculable or observable Book Value. Agents cannot
directly observe V_t. Instead, they only have noisy signals S_{i,t} reflecting
imperfect information about future payoffs.
'''
def generate_fundamental_value(steps=2000, start_price=100, volatility=1.0):
    drift_correction = -0.5 * (volatility ** 2) # https://www.kitces.com/blog/volatility-drag-variance-drain-mean-arithmetic-vs-geometric-average-investment-returns/#:~:text=The%20inevitable%20gap%20between%20the,actual%20geometric%20mean%20of%206.94%25.
    shocks = np.random.normal(loc=drift_correction, scale=volatility, size=steps)
    price_path = start_price * np.exp(np.cumsum(shocks))
    return price_path