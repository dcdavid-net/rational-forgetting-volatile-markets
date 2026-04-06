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

    def _get_representative_return(self, bin_id):
        z_centers = {
            1: -2.5,  # Extreme negative/left tail
            2: -1.25, # Moderate negative center
            3: 0.0,   # Flat center
            4: 1.25,  # Moderate positive center
            5: 2.5    # Extreme positive/right tail
        }
        
        # Calculate the historical volatility of this specific memory chunk, not the current volatility
        memory_tuples = self.memory.get(bin_id, [])
        if not memory_tuples:
            return 0.0
            
        historical_vols = [volatility for _, volatility in memory_tuples]
        avg_historical_vol = sum(historical_vols) / len(historical_vols)
        
        return z_centers[bin_id] * avg_historical_vol

    def observe_return(self, current_return, current_volatility, current_time):
        if current_volatility == 0.0:
            z_score = 0.0
        else:
            z_score = current_return / current_volatility
            
        bin_id = self._get_bin(z_score)

        if bin_id not in self.memory:
            self.memory[bin_id] = []
        self.memory[bin_id].append((current_time, current_volatility)) # (timestamp, volatility_context) tuples

    # def observe_price(self, price, current_time):
    #     if price not in self.memory:
    #         self.memory[price] = []
    #     self.memory[price].append(current_time)

    def _get_contextual_similarity(self, memory_tuples, current_volatility):
        if not memory_tuples:
            return float('-inf')
            
        historical_vols = [volatility for _, volatility in memory_tuples]
        avg_historical_vol = sum(historical_vols) / len(historical_vols)
        
        # so that an 80% drop in volatility yields a massive 0.8 mismatch
        relative_mismatch = abs(current_volatility - avg_historical_vol) / avg_historical_vol
        penalty_scale = 5.0 # scaling the penalty so it can mathematically neutralize the B_i advantage
        mismatch_penalty = -relative_mismatch * penalty_scale
        
        return mismatch_penalty

    def _get_base_level_activation(self, timestamp_list, current_time):
        r'''
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

    def generate_bid_ask_spread(self, current_price, current_volatility, current_time, true_value=None, do_pruning=True, add_noise=True):
        '''
        An agent would have some valuation from its highest-activated memory
        It would then say "I am willing to buy the asset below this price or
        sell it above this price", so we need to generate a bid/ask spread

        Optional 'do_pruning' and 'add_noise' parameters for testing.
        '''
        if not self.memory:
            return None

        context_weight = 1.0 
        
        total_activations = {}
        base_activations = {} # Create a separate dictionary for pruning
        for bin_id, memory_tuples in self.memory.items():
            timestamp_list = [timestamp for timestamp, _ in memory_tuples]
            s_context = self._get_contextual_similarity(memory_tuples, current_volatility)

            b_i = self._get_base_level_activation(timestamp_list, current_time)
            noise = np.random.logistic(loc=0.0, scale=1.0) if add_noise else 0.0
            a_i = b_i + (context_weight * s_context) + noise

            base_activations[bin_id] = b_i
            total_activations[bin_id] = a_i
        
        if do_pruning:
            self._do_prune_memory(base_activations, total_activations)
        
            # check memory again if pruning removed everything
            if not self.memory: 
                return None

        retrieved_bin = max(total_activations, key=total_activations.get)
        r_retrieved = self._get_representative_return(retrieved_bin)
        expected_value = current_price * (1.0 + r_retrieved)
        dynamic_spread = self.spread_pct * current_volatility 
        
        bid_price = expected_value * (1 - (0.5 * dynamic_spread))
        ask_price = expected_value * (1 + (0.5 * dynamic_spread))

        return {
            'agent_id': self.agent_id,
            'bid': bid_price,
            'ask': ask_price
        }