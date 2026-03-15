# Agent Class

import numpy as np
from math import log, pow

class Agent:
    def __init__(self, agent_id, decay_rate=0.5, prune_threshold=-10.0, spread=2.0):
        self.agent_id = agent_id
        self.d = decay_rate
        self.prune_threshold = prune_threshold
        self.spread = spread # static for now. TODO: in Phase 2, may have to make this variable

        self.memory = {}
    
    def __repr__(self):
        return f"Agent({self.agent_id}, decay_rate={self.d}, prune_threshold={self.prune_threshold}, spread={self.spread})"

    def observe_price(self, price, current_time):
        if price not in self.memory:
            self.memory[price] = []
        self.memory[price].append(current_time)

    '''
    An agent would have some valuation from its highest-activated memory
    It would then say "ok I am willing to buy the asset below this price and
    also sell it above this price", so we need to generate a bid/ask spread
    '''
    def generate_orders(self, current_time):
        if not self.memory:
            return None
        
        activations = {}
        for price, timestamp_list in self.memory.items():
            b_i = 0.0
            # TODO: may need to factor this out into its own method
            '''
            CALCULATE BASE-LEVEL ACTIVATION
            $$B_i = \ln \left( \sum_{k=1}^{n} t_k^{-d} \right)$$
            At zero decay, the equation is just the natural log length of timestamp_list
            '''
            if self.d == 0.0: 
                b_i = log(len(timestamp_list))
            else:
                sum_decay = 0.0
                for past_time in timestamp_list:
                    t_k = current_time - past_time
                    
                    '''
                    t_k^{-d} = 1/(t_k^{d}). So if t_k is 0.0, that would be divsion by zero
                    Grounded in Cognitive Theory: 
                    According to Anderson's "An Integrated Theory of the Mind:"
                        <direct quote>
                        "The assumption in ACT-R is that this cycle takes about 50 ms to complete—
                        this estimate of 50 ms as the minimum cycle time for cognition"
                    https://doi.org/10.1037/0033-295x.111.4.1036.
                    '''
                    if t_k > 0.0:
                        sum_decay += pow(t_k, -self.d)
        
                '''
                Natural log only has domain (0,inf]. ln(sum_decay) needs sum_decay to be > 0.0
                Grounded in Cognitive Theory:
                A sum_decay = 0.0 can only happen when t_k = 0.0, which means that we have never encountered
                such item in the past. According to Anderon's "The Atomic Components of Thought:"
                "the odds that an item will be needed are related to its history of past exposure"
                is equivalent to our sum_decay variable. Therefore, an item with sum_decay = 0.0
                is saying that there is zero odds that an item will be needed; and therefore should have totally
                no base-level activation.
                '''
                if sum_decay <= 0.0:
                    b_i = float('-inf')
                else:
                    b_i = log(sum_decay)

            noise = np.random.logistic(loc=0.0, scale=1.0)
            a_i = b_i + noise
            activations[price] = a_i
        
        # TODO: will probably need to prune memory here
        # check memory again if pruning removed everything
        if not self.memory:
            return None

        retrieved_value = max(activations, key=activations.get)