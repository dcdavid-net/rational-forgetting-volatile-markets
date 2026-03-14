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
    shocks = np.random.normal(loc=0, scale=volatility, size=steps)
    price_path = start_price + np.cumsum(shocks)
    return price_path