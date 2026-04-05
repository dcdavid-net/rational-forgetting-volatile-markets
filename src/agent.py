# Agent Class

import numpy as np
from math import log, pow

class Agent:
    def __init__(self, agent_id, decay_rate=0.5, prune_threshold=-2.292483739335286, spread=2.0):
        self.agent_id = agent_id
        self.d = decay_rate
        self.prune_threshold = prune_threshold

        '''
        spread_pct static for now. TODO: in Phase 2, may have to make this variable
        defaulting to 2.0 so that we functionally do spreads of ± 1 percent
        '''
        self.spread_pct = spread

        self.memory = {}
    
    def __repr__(self):
        return f"Agent({self.agent_id}, decay_rate={self.d}, prune_threshold={self.prune_threshold}, spread={self.spread_pct})"

    def _get_bin(self, z_score):
        if z_score < -2.0:
            return 1
        elif z_score < -0.5:
            return 2
        elif z_score <= 0.5:
            return 3
        elif z_score <= 2.0:
            return 4
        else:
            return 5

    def _get_representative_return(self, bin_id, current_volatility):
        z_centers = {
            1: -2.5,  # Extreme negative/left tail
            2: -1.25, # Moderate negative center
            3: 0.0,   # Flat center
            4: 1.25,  # Moderate positive center
            5: 2.5    # Extreme positive/right tail
        }
        return z_centers[bin_id] * current_volatility

    def observe_return(self, current_return, current_volatility, current_time):
        if current_volatility == 0.0:
            z_score = 0.0
        else:
            z_score = current_return / current_volatility
            
        bin_id = self._get_bin(z_score)

        if bin_id not in self.memory:
            self.memory[bin_id] = []
        self.memory[bin_id].append(current_time)

    # def observe_price(self, price, current_time):
    #     if price not in self.memory:
    #         self.memory[price] = []
    #     self.memory[price].append(current_time)

    def _get_base_level_activation(self, timestamp_list, current_time):
        '''
        CALCULATE BASE-LEVEL ACTIVATION
        $$B_i = \ln \left( \sum_{k=1}^{n} t_k^{-d} \right)$$
        '''
        if not timestamp_list:
            return float('-inf')

        past_timestamps = [past_time for past_time in timestamp_list if (current_time - past_time) > 0.0]
        if not past_timestamps:
            return float('-inf')

        # At zero decay, the equation is just the natural log length of timestamp_list
        if self.d == 0.0: 
            return log(len(past_timestamps))
        else:
            sum_decay = 0.0
            for past_time in past_timestamps:
                t_k = current_time - past_time
                sum_decay += pow(t_k, -self.d)
            return log(sum_decay)

    def _do_prune_memory(self, base_activations, total_activations):
        bins_to_prune = [bin_id for bin_id, b_i in base_activations.items() if b_i < self.prune_threshold]
        for bin_id in bins_to_prune:
            del self.memory[bin_id] # delete from memory
            del total_activations[bin_id] # delete from the temporary retrieval memory too

    def generate_bid_ask_spread(self, current_price, current_volatility, current_time, do_pruning=True, add_noise=True):
        '''
        An agent would have some valuation from its highest-activated memory
        It would then say "I am willing to buy the asset below this price or
        sell it above this price", so we need to generate a bid/ask spread

        Optional 'do_pruning' and 'add_noise' parameters for testing.
        '''
        if not self.memory:
            return None
        
        total_activations = {}
        base_activations = {} # Create a separate dictionary for pruning
        for bin_id, timestamp_list in self.memory.items():
            b_i = self._get_base_level_activation(timestamp_list, current_time)
            noise = np.random.logistic(loc=0.0, scale=1.0)
            a_i = b_i + noise if add_noise else b_i

            base_activations[bin_id] = b_i
            total_activations[bin_id] = a_i
        
        if do_pruning:
            self._do_prune_memory(base_activations, total_activations)
        
            # check memory again if pruning removed everything
            if not self.memory: 
                return None

        retrieved_bin = max(total_activations, key=total_activations.get)
        r_retrieved = self._get_representative_return(retrieved_bin, current_volatility)
        expected_value = current_price * (1.0 + r_retrieved)
        bid_price = expected_value * (1 - (0.5 * self.spread_pct / 100))
        ask_price = expected_value * (1 + (0.5 * self.spread_pct / 100))

        return {
            'agent_id': self.agent_id,
            'bid': round(bid_price, 2),
            'ask': round(ask_price, 2)
        }