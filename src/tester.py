# Test all my code
# Has to be called using "python -m src.tester" for it test run main.py

import sys
import os
import numpy as np
from scipy.stats import kurtosis
from main import set_reproducibility_seed
from math import log
from src.generator import generate_fundamental_value
from src.market import Order, Market
from src.agent import Agent

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
    verbose = False
    if len(sys.argv) > 1:
        verbose = '--verbose' in sys.argv
        
    output_file = 'outputs.txt'
    print(f'This tester outputs to {output_file}.\nThere is an optional "--verbose" flag for detailed logs.')

    sys.stdout = Logger(output_file)
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
    print('100 Iterations must have Average Fisher Kurtosis ≈ 0.0 ± 0.05')
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
    print('######################### agent.py ############################')
    print('###############################################################')
    agent = Agent(agent_id=9, decay_rate=0.5, prune_threshold=-10.0, spread=2.0)

    print('Testing Agent memory:')
    print('Newly-initialized agent should have no memory')
    if verbose: print(agent.memory)
    assert agent.memory == {}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('Observed price should be in memory')
    agent.observe_price(price = 100, current_time=11)
    if verbose: print(agent.memory)
    assert agent.memory == {100: [11]}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('New observed price should insert a new price to memory')
    agent.observe_price(price = 102, current_time=12)
    if verbose: print(agent.memory)
    assert agent.memory == {100: [11], 102: [12]}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('Previously observed price should insert to the existing list')
    agent.observe_price(price = 102, current_time=13)
    if verbose: print(agent.memory)
    assert agent.memory == {100: [11], 102: [12,13]}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('Newly-initialized agent should generate no bid_ask_spread from memory')
    agent10 = Agent(agent_id=10, decay_rate=0.5, prune_threshold=-10.0, spread=2.0)
    bid_ask_spread = agent10.generate_bid_ask_spread(current_time=14, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert not bid_ask_spread
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''An empty memory list should have base_level_activation = "-inf"
because the sum_decay would be 0, and ln(0) is undefined''')
    base_level_activation = agent10._get_base_level_activation(timestamp_list=[], current_time=15)
    print(base_level_activation)
    if verbose: print(base_level_activation)
    assert base_level_activation == float('-inf')
    print('Test passed.\n')

    print('######################## Zero Decay ###########################')
    print('---------------------------------------------------------------')
    print('Agent with zero decay and no memory should have base_level_activation of "-inf."')
    agent_zero_decay = Agent(agent_id=10, decay_rate=0.0, prune_threshold=-10.0, spread=2.0)
    base_level_activation = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(100), current_time=15)
    if verbose: print(base_level_activation)
    assert base_level_activation == float('-inf')
    print('Test passed.\n')

    print('Agent with zero decay and no memory should have bid_ask_spread of None')
    bid_ask_spread = agent_zero_decay.generate_bid_ask_spread(current_time=15, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert not bid_ask_spread
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with zero decay and only one memory with one timestamp
should have a base_level_activation of ln(1) = 0.0.''')
    agent_zero_decay.observe_price(price = 100, current_time=14)
    base_level_activation = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(100), current_time=15)
    if verbose: print(f'Agent memory: {agent_zero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation == 0.0
    print('Test passed.\n')

    print('''Agent with zero decay and one memory with one timestamp should have
bid_ask_spread of that one price ± 0.5(spread).''')
    bid_ask_spread = agent_zero_decay.generate_bid_ask_spread(current_time=15, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99, 'ask':101}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with zero decay and only one memory with two timestamps
should have a base_level_activation of ln(21) ≈ 0.69.''')
    agent_zero_decay.observe_price(price = 100, current_time=15)
    base_level_activation = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(100), current_time=16)
    if verbose: print(f'Agent memory: {agent_zero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation == log(2)
    print('Test passed.\n')

    print('''Agent with zero decay and one memory with two timestamps should have
bid_ask_spread of that one price ± 0.5(spread).''')
    bid_ask_spread = agent_zero_decay.generate_bid_ask_spread(current_time=17, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99, 'ask':101}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with zero decay that is trying to get the base_level_activation 
of a "present" price should get "-inf" since the price should not be counted in 
base_level_activation if it is still in the present and not yet in the past.''')
    agent_zero_decay.observe_price(price = 103, current_time=1000000)
    base_level_activation = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(103), current_time=1000000)
    if verbose: print(f'Agent memory: {agent_zero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation == float('-inf')
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with zero decay should indiscriminately pick the price
with the most frequency, regardless of age. Testing price = 100 with 2
instances that are 1,000,000 timesteps old and also price = 103 with 1
instance that is only 1 timestep old.''')
    base_level_activation_100 = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(100), current_time=1000000)
    base_level_activation_103 = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(103), current_time=1000001)
    if verbose: print(f'Agent memory: {agent_zero_decay.memory}. Agent base_level_activation_100 {base_level_activation_100}. Agent base_level_activation_103 {base_level_activation_103}')
    assert (base_level_activation_100, base_level_activation_103) == (log(2), 0.0)
    print('Test passed.\n')

    print('''Agent with zero decay and two prices in its memory should have
bid_ask_spread of the more frequent price ± 0.5(spread).''')
    bid_ask_spread = agent_zero_decay.generate_bid_ask_spread(current_time=1000001, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99, 'ask':101}
    print('Test passed.\n')

    agent_zero_decay.observe_price(price = 103, current_time=1000001)
    agent_zero_decay.observe_price(price = 103, current_time=1000002)
    print('''Agent with zero decay and two prices in its memory should have
bid_ask_spread of the more frequent price ± 0.5(spread).''')
    bid_ask_spread = agent_zero_decay.generate_bid_ask_spread(current_time=1000003, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':103*0.99, 'ask':103*1.01}
    print('Test passed.\n')

    print('###################### Non-Zero Decay #########################')
    print('---------------------------------------------------------------')
    print('Agent with non-zero decay and no memory should have base_level_activation of "-inf."')
    agent_nonzero_decay = Agent(agent_id=10, decay_rate=0.5, prune_threshold=-10.0, spread=2.0)
    base_level_activation = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(100), current_time=15)
    if verbose: print(base_level_activation)
    assert base_level_activation == float('-inf')
    print('Test passed.\n')

    print('Agent with non-zero decay and no memory should have bid_ask_spread of None')
    bid_ask_spread = agent_nonzero_decay.generate_bid_ask_spread(current_time=15, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert not bid_ask_spread
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with non-zero decay and only one memory with one timestamp
should have a base_level_activation of ln(1) = 0.0.''')
    agent_nonzero_decay.observe_price(price = 100, current_time=14)
    base_level_activation = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(100), current_time=15)
    if verbose: print(f'Agent memory: {agent_nonzero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation == 0.0
    print('Test passed.\n')

    print('''Agent with non-zero decay and one memory with one timestamp should have
bid_ask_spread of that one price ± 0.5(spread).''')
    bid_ask_spread = agent_nonzero_decay.generate_bid_ask_spread(current_time=15, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99, 'ask':101}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with non-zero decay and only one memory with two timestamps
should have a base_level_activation of less than ln(21) ≈ 0.69.''')
    # Personal reminder:
    # Would it truly ALWAYS be less than?
    # Noise is additive, so... couldnt total activation be sometimes greater than ln(length of list)?
    # Well no, this is base level activation b_i. Noise is in total activation a_i.
    agent_nonzero_decay.observe_price(price = 100, current_time=15)
    base_level_activation = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(100), current_time=16)
    if verbose: print(f'Agent memory: {agent_nonzero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation < log(2)
    print('Test passed.\n')

    print('''Agent with non-zero decay and one memory with two timestamps should have
bid_ask_spread of that one price ± 0.5(spread).''')
    bid_ask_spread = agent_nonzero_decay.generate_bid_ask_spread(current_time=17, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99, 'ask':101}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with non-zero decay that is trying to get the base_level_activation 
of a "present" price should get "-inf" since the price should not be counted in 
base_level_activation if it is still in the present and not yet in the past.''')
    agent_nonzero_decay.observe_price(price = 103, current_time=1000000)
    base_level_activation = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(103), current_time=1000000)
    if verbose: print(f'Agent memory: {agent_nonzero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation == float('-inf')
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with non-zero decay should pick the price that is a lot more recent, 
holding less regard to frequency than a zero-decay agent. Testing price = 100 with 2
instances that are 1,000,000 timesteps old and also price = 103 with 1
instance that is only 1 timestep old.''')
    base_level_activation_100 = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(100), current_time=1000000)
    base_level_activation_103 = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(103), current_time=1000001)
    if verbose: print(f'Agent memory: {agent_nonzero_decay.memory}. Agent base_level_activation_100 {base_level_activation_100}. Agent base_level_activation_103 {base_level_activation_103}')
    assert base_level_activation_100 < base_level_activation_103
    print('Test passed.\n')

    print('''Agent with non-zero decay and two prices in its memory should have
bid_ask_spread of the more recent price ± 0.5(spread).''')
    bid_ask_spread = agent_nonzero_decay.generate_bid_ask_spread(current_time=1000001, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread) 
    assert bid_ask_spread == {'agent_id':10, 'bid':103*0.99, 'ask':103*1.01}
    print('Test passed.\n')

    agent_nonzero_decay.observe_price(price = 100, current_time=14)
    agent_nonzero_decay.observe_price(price = 100, current_time=15)
    agent_nonzero_decay.observe_price(price = 103, current_time=1000000)
    agent_nonzero_decay.observe_price(price = 100, current_time=1000001)
    agent_nonzero_decay.observe_price(price = 100, current_time=1000002)
    print('''Agent with non-zero decay and two prices in its memory should have
bid_ask_spread of the more recent price ± 0.5(spread).''')
    bid_ask_spread = agent_nonzero_decay.generate_bid_ask_spread(current_time=1000003, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99, 'ask':101}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    # t_1 = 1,000,000,000 - 14 ≈ 1,000,000,000
    # t_2 = 1,000,000,000 - 15 ≈ 1,000,000,000
    # t_1^{-0.5} ≈ 0.00003162277
    # t_2^{-0.5} ≈ 0.00003162277
    # sum t_1, t_2 ≈ 0.00006324554
    # ln(sum t_2, t_2) ≈ -9.66848594668 < -10 pruning threshold
    # so gotta have 1,000,000,000,000 or 1 trillion instead of just 1 billion
    agent_nonzero_decay_with_pruning = Agent(agent_id=10, decay_rate=0.5, prune_threshold=-10.0, spread=2.0)
    agent_nonzero_decay_with_pruning.observe_price(price = 100, current_time=14)
    agent_nonzero_decay_with_pruning.observe_price(price = 100, current_time=15)
    agent_nonzero_decay_with_pruning.observe_price(price = 103, current_time=1000000000)
    print('''Agent with non-zero decay should prune very old prices.''')
    bid_ask_spread = agent_nonzero_decay_with_pruning.generate_bid_ask_spread(current_time=1000000001, do_pruning=True, add_noise=False)
    if verbose: print(agent_nonzero_decay_with_pruning.memory)
    assert agent_nonzero_decay_with_pruning.memory == {100: [14, 15], 103: [1000000000]}
    print('Test passed.\n')

    print('''Agent with non-zero decay should prune very old prices.''')
    agent_nonzero_decay_with_pruning.observe_price(price = 103, current_time=1000000000000) # to prevent pruning of this price
    bid_ask_spread = agent_nonzero_decay_with_pruning.generate_bid_ask_spread(current_time=1000000000001, do_pruning=True, add_noise=False)
    if verbose: print(agent_nonzero_decay_with_pruning.memory)
    assert agent_nonzero_decay_with_pruning.memory == {103: [1000000000, 1000000000000]}
    print('Test passed.\n')

    # print('---------------------------------------------------------------')
    # print('Template')
    # var = 1
    # if verbose: print(f'')
    # assert 1
    # print('Test passed.\n')