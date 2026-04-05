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
    print('Observed return should be chunked into the correct bin and stored in memory')
    # Z-score = 0.0 / 0.01 = 0.0 -> Translates to Bin 3 (Flat/Noise)
    agent.observe_return(current_return=0.0, current_volatility=0.01, current_time=11)
    if verbose: print(agent.memory)
    assert agent.memory == {3:[11]}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('New observed return in a different category should insert a new bin ID to memory')
    # Z-score = 0.01 / 0.01 = 1.0 -> Translates to Bin 4 (Moderate Positive)
    agent.observe_return(current_return=0.01, current_volatility=0.01, current_time=12)
    if verbose: print(agent.memory)
    assert agent.memory == {3:[11], 4:[12]}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('Previously observed category should append the timestamp to the existing bin list')
    # Z-score = 0.015 / 0.01 = 1.5 -> Translates to Bin 4 (Moderate Positive)
    agent.observe_return(current_return=0.015, current_volatility=0.01, current_time=13)
    if verbose: print(agent.memory)
    assert agent.memory == {3:[11], 4:[12,13]}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('Newly-initialized agent should generate no bid_ask_spread from memory')
    agent10 = Agent(agent_id=10, decay_rate=0.5, prune_threshold=-10.0, spread=2.0)
    # We pass in dummy values (100.0 and 0.01) just to satisfy the new method signature
    bid_ask_spread = agent10.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=14, do_pruning=False, add_noise=False)
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
    # Bin 3 (Flat/Noise) as our standard bin instead of "price 100"
    base_level_activation = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(3), current_time=15)
    if verbose: print(base_level_activation)
    assert base_level_activation == float('-inf')
    print('Test passed.\n')

    print('Agent with zero decay and no memory should have bid_ask_spread of None')
    bid_ask_spread = agent_zero_decay.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=15, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert not bid_ask_spread
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with zero decay and only one memory with one timestamp
should have a base_level_activation of ln(1) = 0.0.''')
    # z_score = 0.0 / 0.01 = 0.0 (Bin 3)
    agent_zero_decay.observe_return(current_return=0.0, current_volatility=0.01, current_time=14)
    base_level_activation = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(3), current_time=15)
    if verbose: print(f'Agent memory: {agent_zero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation == 0.0
    print('Test passed.\n')

    print('''Agent with zero decay and one memory with one timestamp should have
bid_ask_spread centered around expected value.''')
    # Retrieved Bin 3 (z_center = 0.0) -> Expected Return = 0.0 * 0.01 = 0.0 -> Expected Value = 100 * (1 + 0) = 100
    bid_ask_spread = agent_zero_decay.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=15, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99.0, 'ask':101.0}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with zero decay and only one memory with two timestamps
should have a base_level_activation of ln(2) ≈ 0.69.''')
    agent_zero_decay.observe_return(current_return=0.0, current_volatility=0.01, current_time=15)
    base_level_activation = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(3), current_time=16)
    if verbose: print(f'Agent memory: {agent_zero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation == log(2)
    print('Test passed.\n')

    print('''Agent with zero decay and one memory with two timestamps should have
bid_ask_spread centered around expected value.''')
    bid_ask_spread = agent_zero_decay.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=17, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99.0, 'ask':101.0}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with zero decay that is trying to get the base_level_activation 
of a "present" return should get "-inf" since the return should not be counted in 
base_level_activation if it is still in the present and not yet in the past.''')
    # Simulate a moderate positive return: z_score = 0.015 / 0.01 = 1.5 (Bin 4)
    agent_zero_decay.observe_return(current_return=0.015, current_volatility=0.01, current_time=1000000)
    base_level_activation = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(4), current_time=1000000)
    if verbose: print(f'Agent memory: {agent_zero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation == float('-inf')
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with zero decay should indiscriminately pick the bin
with the most frequency, regardless of age. Testing Bin 3 with 2
instances that are 1,000,000 timesteps old and also Bin 4 with 1
instance that is only 1 timestep old.''')
    base_level_activation_3 = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(3), current_time=1000000)
    base_level_activation_4 = agent_zero_decay._get_base_level_activation(timestamp_list=agent_zero_decay.memory.get(4), current_time=1000001)
    if verbose: print(f'Agent memory: {agent_zero_decay.memory}. Agent base_level_activation_3 {base_level_activation_3}. Agent base_level_activation_4 {base_level_activation_4}')
    assert (base_level_activation_3, base_level_activation_4) == (log(2), 0.0)
    print('Test passed.\n')

    print('''Agent with zero decay and two bins in its memory should have
bid_ask_spread centered around the expected value of the more frequent bin.''')
    # retrieve Bin 3 (z_center = 0.0) -> Expected Return = 0.0
    bid_ask_spread = agent_zero_decay.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=1000001, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99.0, 'ask':101.0}
    print('Test passed.\n')

    # add two more Bin 4 instances so it overtakes Bin 3 in frequency
    agent_zero_decay.observe_return(current_return=0.015, current_volatility=0.01, current_time=1000001)
    agent_zero_decay.observe_return(current_return=0.015, current_volatility=0.01, current_time=1000002)
    print('''Agent with zero decay and two bins in its memory should have
bid_ask_spread centered around the expected value of the more frequent bin.''')
    # retrieve Bin 4 (z_center = 1.25) -> Expected Return = 1.25 * 0.01 = 0.0125
    # Expected Value = 100 * (1 + 0.0125) = 101.25
    # Spread = +/- 1% -> Bid = 100.24, Ask = 102.26
    bid_ask_spread = agent_zero_decay.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=1000003, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':100.24, 'ask':102.26}
    print('Test passed.\n')

    print('###################### Non-Zero Decay #########################')
    print('---------------------------------------------------------------')
    print('Agent with non-zero decay and no memory should have base_level_activation of "-inf."')
    agent_nonzero_decay = Agent(agent_id=10, decay_rate=0.5, prune_threshold=-10.0, spread=2.0)
    base_level_activation = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(3), current_time=15)
    if verbose: print(base_level_activation)
    assert base_level_activation == float('-inf')
    print('Test passed.\n')

    print('Agent with non-zero decay and no memory should have bid_ask_spread of None')
    bid_ask_spread = agent_nonzero_decay.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=15, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert not bid_ask_spread
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with non-zero decay and only one memory with one timestamp
should have a base_level_activation of ln(1) = 0.0.''')
    # z_score = 0.0 -> Bin 3
    agent_nonzero_decay.observe_return(current_return=0.0, current_volatility=0.01, current_time=14)
    base_level_activation = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(3), current_time=15)
    if verbose: print(f'Agent memory: {agent_nonzero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation == 0.0
    print('Test passed.\n')

    print('''Agent with non-zero decay and one memory with one timestamp should have
bid_ask_spread centered around expected value.''')
    # Retrives Bin 3 (z_center = 0.0) -> Expected return 0.0 -> Expected Value = 100.0
    bid_ask_spread = agent_nonzero_decay.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=15, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99.0, 'ask':101.0}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with non-zero decay and only one memory with two timestamps
should have a base_level_activation of less than ln(2) ≈ 0.69.''')
    # Personal reminder:
    # Would it truly ALWAYS be less than?
    # Noise is additive, so... couldnt total activation be sometimes greater than ln(length of list)?
    # Well no, this is base level activation b_i. Noise is in total activation a_i.
    agent_nonzero_decay.observe_return(current_return=0.0, current_volatility=0.01, current_time=15)
    base_level_activation = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(3), current_time=16)
    if verbose: print(f'Agent memory: {agent_nonzero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation < log(2)
    print('Test passed.\n')

    print('''Agent with non-zero decay and one memory with two timestamps should have
bid_ask_spread centered around expected value.''')
    bid_ask_spread = agent_nonzero_decay.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=17, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99.0, 'ask':101.0}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with non-zero decay that is trying to get the base_level_activation 
of a "present" return should get "-inf" since the return should not be counted in 
base_level_activation if it is still in the present and not yet in the past.''')
    # z_score = 0.015 / 0.01 = 1.5 -> Bin 4
    agent_nonzero_decay.observe_return(current_return=0.015, current_volatility=0.01, current_time=1000000)
    base_level_activation = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(4), current_time=1000000)
    if verbose: print(f'Agent memory: {agent_nonzero_decay.memory}. Agent base_level_activation {base_level_activation}')
    assert base_level_activation == float('-inf')
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    print('''Agent with non-zero decay should pick the bin that is a lot more recent, 
holding less regard to frequency than a zero-decay agent. Testing Bin 3 with 2
instances that are 1,000,000 timesteps old and also Bin 4 with 1
instance that is only 1 timestep old.''')
    base_level_activation_3 = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(3), current_time=1000000)
    base_level_activation_4 = agent_nonzero_decay._get_base_level_activation(timestamp_list=agent_nonzero_decay.memory.get(4), current_time=1000001)
    if verbose: print(f'Agent memory: {agent_nonzero_decay.memory}. Agent base_level_activation_3 {base_level_activation_3}. Agent base_level_activation_4 {base_level_activation_4}')
    assert base_level_activation_3 < base_level_activation_4
    print('Test passed.\n')

    print('''Agent with non-zero decay and two bins in its memory should have
bid_ask_spread based on the expected return of the more recent bin.''')
    # Retrieves Bin 4 (z_center = 1.25) -> Expected Return = 1.25 * 0.01 = 0.0125
    # Expected Value = 100 * (1 + 0.0125) = 101.25
    # Spread = +/- 1% -> Bid = 100.24, Ask = 102.26
    bid_ask_spread = agent_nonzero_decay.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=1000001, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread) 
    assert bid_ask_spread == {'agent_id':10, 'bid':100.24, 'ask':102.26}
    print('Test passed.\n')

    # Load up Bin 3 so its combined base-level activation overcomes the recent Bin 4
    agent_nonzero_decay.observe_return(current_return=0.0, current_volatility=0.01, current_time=14)
    agent_nonzero_decay.observe_return(current_return=0.0, current_volatility=0.01, current_time=15)
    agent_nonzero_decay.observe_return(current_return=0.015, current_volatility=0.01, current_time=1000000) # Bin 4
    agent_nonzero_decay.observe_return(current_return=0.0, current_volatility=0.01, current_time=1000001)
    agent_nonzero_decay.observe_return(current_return=0.0, current_volatility=0.01, current_time=1000002)
    print('''Agent with non-zero decay and two bins in its memory should have
bid_ask_spread based on the expected return of the highly frequent & recent bin.''')
    # Now Bin 3 takes over again. Expected value = 100.0
    bid_ask_spread = agent_nonzero_decay.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=1000003, do_pruning=False, add_noise=False)
    if verbose: print(bid_ask_spread)
    assert bid_ask_spread == {'agent_id':10, 'bid':99.0, 'ask':101.0}
    print('Test passed.\n')

    print('---------------------------------------------------------------')
    # t_1 = 1,000,000,000 - 14 ≈ 1,000,000,000
    # t_2 = 1,000,000,000 - 15 ≈ 1,000,000,000
    # t_1^{-0.5} ≈ 0.00003162277
    # t_2^{-0.5} ≈ 0.00003162277
    # sum t_1, t_2 ≈ 0.00006324554
    # ln(sum t_2, t_2) ≈ -9.66848594668 > -10 pruning threshold
    # so gotta have 1,000,000,000,000 or 1 trillion instead of just 1 billion
    agent_nonzero_decay_with_pruning = Agent(agent_id=10, decay_rate=0.5, prune_threshold=-10.0, spread=2.0)
    agent_nonzero_decay_with_pruning.observe_return(current_return=0.0, current_volatility=0.01, current_time=14)
    agent_nonzero_decay_with_pruning.observe_return(current_return=0.0, current_volatility=0.01, current_time=15)
    agent_nonzero_decay_with_pruning.observe_return(current_return=0.015, current_volatility=0.01, current_time=1000000000)
    print('''Agent with non-zero decay should NOT prune moderately old prices yet.''')
    bid_ask_spread = agent_nonzero_decay_with_pruning.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=1000000001, do_pruning=True, add_noise=False)
    if verbose: print(agent_nonzero_decay_with_pruning.memory)
    assert agent_nonzero_decay_with_pruning.memory == {3:[14,15], 4:[1000000000]}
    print('Test passed.\n')

    print('''Agent with non-zero decay should prune extremely old prices.''')
    agent_nonzero_decay_with_pruning.observe_return(current_return=0.015, current_volatility=0.01, current_time=1000000000000) # to prevent pruning of Bin 4
    bid_ask_spread = agent_nonzero_decay_with_pruning.generate_bid_ask_spread(current_price=100.0, current_volatility=0.01, current_time=1000000000001, do_pruning=True, add_noise=False)
    if verbose: print(agent_nonzero_decay_with_pruning.memory)
    # Bin 3 (return 0.0) from t=14 and t=15 finally decays past -10.0 threshold and is deleted
    assert agent_nonzero_decay_with_pruning.memory == {4:[1000000000,1000000000000]}
    print('Test passed.\n')

    # print('---------------------------------------------------------------')
    # print('Template')
    # var = 1
    # if verbose: print(f'')
    # assert 1
    # print('Test passed.\n')