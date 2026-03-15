# Agent Class

import numpy as np
from math import log, pow

class Agent:
    def __init__(self, agent_id, decay_rate=0.5, prune_threshold=-4.0, spread=2.0):
        '''
        prune_threshold at -10.0 would have taken 500M days to erase a once-seen price.
        Aside from prune_threshold, the other two levers are decay rate and number of timesteps per experiment.
        1. Decay Rate = 0.5 has biological grounding according to “ACT-R 7.30 + Reference Manual”
        2. 500M timesteps would be computationally intensive that would not be quite possible for my machine.
        3. Therefore, we adjust prune_threshold at [TODO: get the empirical threshold] would have taken 2980 days or 11.8 years to erase a once-seen price.
            - TODO: Empirical reverse-engineering

        According to Anderson's "The Atomic Components of Thought," it is necessary "to fit this data
        [by estimating the prune_threshold]... with the decay rate d fixed at .5." 
        Rather than an arbitrary -4.0 or -10.0, we reverse-engineer the prune_threshold using empirical 
        S&P 500 data. It is calibrated so the base-level activation of an extreme event (e.g., a Bin 1 crash) 
        barely survives the maximum historical time gap between major market shocks (e.g., 2008 to 2020). 
        This prevents premature systemic amnesia while managing computational constraints.

        Important counterargument:
        According to Malmendier's "Depression Babies: Do Macroeconomic Experiences Affect Risk-Taking?,"
        "individuals who have experienced low stock-market returns throughout their lives report lower
        willingness to take financial risks." This sounds contrary to how this IBL agent works since
        we are weighing more frequent and more recent experiences more than early-life experiences.

        However, based on that same paper by Malmendier, "more recent returns experiences have stronger
        effects, but experiences early in life still have significant influence." Also, according to 
        Jiang's "Investor Memory and Biased Beliefs: Evidence from the Field," "when prompted to recall
        a past market episode, investors tend to retrieve both recent episodes and distant episodes
        featuring dramatic market movements such as bubbles and crashes."
        
        This is exactly how our IBL agent works because early-life experiences that have bundled 
        extreme returns would have a high activation due to the summation of the base level activation.

        As a result of this literature review, we should actually be storing returns rather than continuous prices.
        - Continuous prices run into a problem where a price like 100.01 and 100.02 do not have shared activation.
        This is Representational Sparsity, where the representation of what an agent stores in its memory creates too
        much sparsity in activation.
        - Binned prices partially solves the problem by grouping continuous prices into intervals like $0.25. In that
        instance, $100.01 and $100.24 would be grouped together. However, $100.24 and $100.26 are closer together
        than $100.01 and $100.24. This is Quantization Error, where binning continuous values creates a stair-step
        pattern that limits precision.
        - Instead, storing binned returns via rolling Z-scores (standard deviations from the mean) allows for 
        dynamic, environment-relative categorization:
            - Bin 1: Extreme Negative (< -2σ)
            - Bin 2: Moderate Negative (-2σ to -0.5σ)
            - Bin 3: Flat / Noise (-0.5σ to 0.5σ)
            - Bin 4: Moderate Positive (0.5σ to 2σ)
            - Bin 5: Extreme Positive (> 2σ)
            
        Why Z-Scores? According to Anderson's "Is human cognition adaptive?," "A rational analysis is an explanation 
        of an aspect of human behavior based on the assumption that it is optimized somehow to the structure of 
        the environment... the strong regularities in basic cognitive functions such as memory or categorization 
        reflect a statistical optimization to the structure of the environment." 
        
        Since our Fundamental Value is a Gaussian Random Walk, our agents natively expect a normal distribution. 
        However, in real financial markets, according to Welch's "A Heuristic for Fat-TailedStock Market Returns,"
        "Large negative stock and equity portfolio rates of return occur more frequently than they should under 
        the Gaussian normal distribution. They tend to be kurtotic."
        
        This dynamically models human "Normalcy Bias"—our agents systematically expect a standard bell curve, 
        so they are completely blindsided by fat-tail events. This cognitive flaw acts as the exact psychological 
        catalyst for the endogenous crashes observed in our Phase 1 outputs.

        In ACT-R and IBL, Total Activation A_i includes Contextual Similarity S_{context}:
            equation: A_i = B_i + S_{context} + \epsilon
        While Bin 3 dominates B_i, distant dramatic episodes (Bins 1 and 5) are retrieved when the current market
        context (e.g., sudden EWMA volatility spike) matches the historical context of extreme bins.
        The similarity match provides a massive context boost, overriding the high base-level activation of Bin 3.
        This triggers the recall of distant market shocks, completing the feedback loop of panic.
        '''
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
        return f"Agent({self.agent_id}, decay_rate={self.d}, prune_threshold={self.prune_threshold}, spread={self.spread})"

    def observe_price(self, price, current_time):
        if price not in self.memory:
            self.memory[price] = []
        self.memory[price].append(current_time)

    def _get_base_level_activation(self, timestamp_list, current_time):
        '''
        CALCULATE BASE-LEVEL ACTIVATION
        $$B_i = \ln \left( \sum_{k=1}^{n} t_k^{-d} \right)$$
        '''

        '''
        Why past_timestamps?
        t_k^{-d} = 1/(t_k^{d}). So if t_k or (current_time - t) is 0.0, that would be divsion by zero
        Grounded in Cognitive Theory: 
        According to Anderson's "An Integrated Theory of the Mind:"
            <direct quote>
            "The assumption in ACT-R is that this cycle takes about 50 ms to complete—
            this estimate of 50 ms as the minimum cycle time for cognition"
        https://doi.org/10.1037/0033-295x.111.4.1036.

        Why float('-inf')?
        Natural log only has domain (0,inf]. ln(sum_decay) needs sum_decay to be > 0.0
        Grounded in Cognitive Theory:
        A sum_decay = 0.0 can only happen when t_k = 0.0, which means that we have never encountered
        such item in the past. According to Anderon's "The Atomic Components of Thought:"
        "the odds that an item will be needed are related to its history of past exposure"
        is equivalent to our sum_decay variable. Therefore, an item with sum_decay = 0.0
        is saying that there is zero odds that an item will be needed; and therefore should have totally
        no base-level activation.
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

    def _do_prune_memory(self, activations):
        prices_to_prune = [price for price, a_i in activations.items() if a_i < self.prune_threshold]
        for price in prices_to_prune:
            del self.memory[price]

    def generate_bid_ask_spread(self, current_time, do_pruning=True, add_noise=True):
        '''
        An agent would have some valuation from its highest-activated memory
        It would then say "I am willing to buy the asset below this price or
        sell it above this price", so we need to generate a bid/ask spread

        Optional 'do_pruning' and 'add_noise' parameters for testing.
        '''
        if not self.memory:
            return None
        
        activations = {}
        for price, timestamp_list in self.memory.items():
            b_i = self._get_base_level_activation(timestamp_list, current_time)
            noise = np.random.logistic(loc=0.0, scale=1.0)
            a_i = b_i + noise if add_noise else b_i
            activations[price] = a_i
        
        if do_pruning:
            self._do_prune_memory(activations)
        
            # check memory again if pruning removed everything
            if not self.memory: 
                return None

        retrieved_value = max(activations, key=activations.get)

        bid_price = retrieved_value * (1 - (0.5 * self.spread_pct / 100))
        ask_price = retrieved_value * (1 + (0.5 * self.spread_pct / 100))

        return {
            'agent_id': self.agent_id,
            'bid': round(bid_price, 2),
            'ask': round(ask_price, 2)
        }