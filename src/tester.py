# Test all my code
# Has to be called using "python -m src.tester" for it test run main.py

import sys
import os
import numpy as np
from scipy.stats import kurtosis
from main import set_reproducibility_seed
from src.generator import generate_fundamental_value
from src.market import Order, Market
from src.agent import Agent

verbose = False

class Logger(object):
    def __init__(self, filename="outputs.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def prediction_vs_actual(actual_asks_or_bids_object, predicted_asks_or_bids_list, custom_sorting=None):
    actual_values = []
    for obj in actual_asks_or_bids_object:
        actual_values.append((obj.agent_id, obj.order_type, obj.price, obj.timestamp))
    
    if custom_sorting:
        actual_values = sorted(actual_values, key=lambda x: x[custom_sorting])

    for idx, av in enumerate(actual_values):
        pv = actual_values[idx]
        assert pv == av
    return

if __name__ == '__main__':
    sys.stdout = Logger("outputs.txt")
    sys.stderr = sys.stdout

    print('###############################################################')
    print('########################### main.py ###########################')
    print('###############################################################')
    print('Testing Reproducibility:')
    print('100 Iterations must be Equal (main.py)')

    set_reproducibility_seed(0)
    prev_price_path = generate_fundamental_value(steps=2000, start_price=100, volatility=1.0)
    for i in range(100):
        set_reproducibility_seed(0) # reset the seed
        curr_price_path = generate_fundamental_value(steps=2000, start_price=100, volatility=1.0)

        if verbose:
            print(f'Successfully generated {len(curr_price_path)} price steps.')
            print(f'First 5 prices: {curr_price_path[:5]}')
            print(f'Final price: {curr_price_path[-1]:.2f}')

        assert np.array_equal(prev_price_path, curr_price_path)
        prev_price_path = curr_price_path # set current to prev
    print('Test passed.\n')

    print('###############################################################')
    print('######################## generator.py #########################')
    print('###############################################################')
    print('Testing Gaussian Random Walk:')
    print('100 Iterations must have Average Fisher Kurtosis ~ 0.0 ± 0.05')
    set_reproducibility_seed(0) # reset the seed
    fisher_kurtoses = []
    for i in range(100):
        price_path = generate_fundamental_value(steps=2000, start_price=100, volatility=1.0)
        returns = np.diff(price_path)
        fisher_kurtosis = kurtosis(returns, fisher=True)

        if verbose:
            print("Fisher Kurtosis (expected around 0.0 for normal distribution):", fisher_kurtosis)
        
        fisher_kurtoses.append(fisher_kurtosis)
    assert -0.05 <= np.mean(fisher_kurtoses) <= 0.05
    print('Test passed.\n')

    print('###############################################################')
    print('######################### market.py ###########################')
    print('###############################################################')
    print('Testing Order Book timestamps:')
    print('First order placed should be lowest timestamp')
    market = Market()

    # Lets populate the Ask side (Sellers)
    market.submit_order(order_type='ask', agent_id=1, price=105)
    market.submit_order(order_type='ask', agent_id=99, price=102) # First 102 + weird agent_id to test no side effects
    market.submit_order(order_type='ask', agent_id=2, price=102) # Second 102, to test edge case

    predicted_ask_values = [
        (1,'ask',105,1),
        (99,'ask',102,2),
        (2,'ask',102,3)
    ]
    if verbose: print(f'Asks: {market.asks}')
    prediction_vs_actual(market.asks, predicted_ask_values, 3) # element 3 is the timestamp
    print('Test passed.\n')
    
    print('---------------------------------------------------------------')
    print('Testing Order Book ordering asks:')
    print('The first ask should be the lowest price first, then oldest')
    predicted_ask_values = [
        (99,'ask',102,2),
        (2,'ask',102,3),
        (1,'ask',105,1)
    ]
    if verbose: print(f'Asks: {market.asks}')
    prediction_vs_actual(market.asks, predicted_ask_values)
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('Testing Order Book ordering bids:')
    print('The first ask should be the highest price first, then oldest')
    # Lets populate the Bid side (Buyers)
    market.submit_order(order_type='bid', agent_id=3, price=98)
    market.submit_order(order_type='bid', agent_id=100, price=100) # First 100 + weird agent_id to test no side effects
    market.submit_order(order_type='bid', agent_id=4, price=100) # Second 100, to test edge case

    predicted_bid_values = [
        (100,'bid',100,5),
        (4,'bid',100,6),
        (3,'bid',98,4)
    ]
    if verbose: print(f'Bids: {market.bids}')
    prediction_vs_actual(market.bids, predicted_bid_values)
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('Testing Order Book trade executions:')
    print('''There should be no executed trades yet
    since the Asks and Bids have not yet intersected''')
    if verbose: print(f'Trade History: {market.trade_history}')
    assert not market.trade_history
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('Testing Order Book trade executions:')
    print('''Placing a Bid that EXACTLY matches the lowest and oldest Ask
    should execute that Bid and that lowest + oldest Ask at that price''')
    print(f'Placing a Bid order at $102... while Asks are {market.asks}')
    executed_trade = market.submit_order(order_type='bid', agent_id=5, price=102)
    actual_executed_trade = (executed_trade[0]['buyer_id'], executed_trade[0]['seller_id'], executed_trade[0]['price'], executed_trade[0]['timestamp'])
    predicted_executed_trade = (5, 99, 102, 7)
    if verbose: print(actual_executed_trade, predicted_executed_trade)
    assert actual_executed_trade == predicted_executed_trade
    print('Test passed (executed trade).')

    predicted_ask_values = [
        (2,'ask',102,3),
        (1,'ask',105,1)
    ]
    if verbose: print(f'Asks: {market.asks}')
    prediction_vs_actual(market.asks, predicted_ask_values)
    print('Test passed (lowest and oldest Ask is removed from the order book).')

    predicted_bid_values = [
        (100,'bid',100,5),
        (4,'bid',100,6),
        (3,'bid',98,4)
    ]
    if verbose: print(f'Bids: {market.bids}')
    prediction_vs_actual(market.bids, predicted_bid_values)
    print('Test passed (all the same Bids are still in the order book).\n')

    print('---------------------------------------------------------------')
    print('Testing Order Book trade executions:')
    print('''Placing a Bid that is HIGHER than the lowest and oldest Ask
    should execute that Bid and that lowest + oldest Ask at the resting price
    which would be the price of the lowest and oldest Ask''')
    print(f'Placing a Bid order at $105... while Asks are {market.asks}')
    executed_trade = market.submit_order(order_type='bid', agent_id=6, price=105)
    actual_executed_trade = (executed_trade[0]['buyer_id'], executed_trade[0]['seller_id'], executed_trade[0]['price'], executed_trade[0]['timestamp'])
    predicted_executed_trade = (6, 2, 102, 8)
    if verbose: print(actual_executed_trade, predicted_executed_trade)
    assert actual_executed_trade == predicted_executed_trade
    print('Test passed (executed trade).')

    predicted_ask_values = [
        (1,'ask',105,1)
    ]
    if verbose: print(f'Asks: {market.asks}')
    prediction_vs_actual(market.asks, predicted_ask_values)
    print('Test passed (lowest and oldest Ask is removed from the order book).')

    predicted_bid_values = [
        (100,'bid',100,5),
        (4,'bid',100,6),
        (3,'bid',98,4)
    ]
    if verbose: print(f'Bids: {market.bids}')
    prediction_vs_actual(market.bids, predicted_bid_values)
    print('Test passed (all the same Bids are still in the order book).\n')

    print('---------------------------------------------------------------')
    print('Testing Order Book trade executions:')
    print('''Placing an Ask that EXACTLY matches the highest and oldest Bid
    should execute that Ask and that highest + oldest Bid at that price''')
    print(f'Placing an Ask order at $100... while Bids are {market.bids}')
    executed_trade = market.submit_order(order_type='ask', agent_id=7, price=100)
    actual_executed_trade = (executed_trade[0]['buyer_id'], executed_trade[0]['seller_id'], executed_trade[0]['price'], executed_trade[0]['timestamp'])
    predicted_executed_trade = (100, 7, 100, 9)
    if verbose: print(f'Executed trade {executed_trade}')
    assert actual_executed_trade == predicted_executed_trade
    print('Test passed (executed trade).')

    predicted_ask_values = [
        (1,'ask',105,1)
    ]
    if verbose: print(f'Asks: {market.asks}')
    prediction_vs_actual(market.asks, predicted_ask_values)
    print('Test passed (all the same Asks are still in the order book).')

    predicted_bid_values = [
        (4,'bid',100,6),
        (3,'bid',98,4)
    ]
    if verbose: print(f'Bids: {market.bids}')
    prediction_vs_actual(market.bids, predicted_bid_values)
    print('Test passed (highest and oldest Bid is removed from the order book).\n')

    print('---------------------------------------------------------------')
    print('Testing Order Book trade executions:')
    print('''Placing an Ask that is LOWER than the highest and oldest Bid
    should execute that Ask and that highest + oldest Bid at the resting price
    which would be the price of the highest and oldest Bid''')
    print(f'Placing an Ask order at $98... while Bids are {market.bids}')
    executed_trade = market.submit_order(order_type='ask', agent_id=8, price=98)
    actual_executed_trade = (executed_trade[0]['buyer_id'], executed_trade[0]['seller_id'], executed_trade[0]['price'], executed_trade[0]['timestamp'])
    predicted_executed_trade = (4, 8, 100, 10)
    if verbose: print(actual_executed_trade, predicted_executed_trade)
    assert actual_executed_trade == predicted_executed_trade
    print('Test passed (executed trade).')

    predicted_ask_values = [
        (1,'ask',105,1)
    ]
    if verbose: print(f'Asks: {market.asks}')
    prediction_vs_actual(market.asks, predicted_ask_values)
    print('Test passed (all the same Asks are still in the order book).')

    predicted_bid_values = [
        (3,'bid',98,4)
    ]
    if verbose: print(f'Bids: {market.bids}')
    prediction_vs_actual(market.bids, predicted_bid_values)
    print('Test passed (highest and oldest Bid is removed from the order book).\n')

    print('###############################################################')
    print('######################### market.py ###########################')
    print('###############################################################')
    print('Testing Agent memory:')
    print('Observed price should be in memory')
    agent = Agent(agent_id=9, decay_rate=0.5, prune_threshold=-10.0, spread=2.0)
    agent.observe_price(price = 100, current_time=11)
    