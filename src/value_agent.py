class ValueAgent:
    def __init__(self, agent_id, spread=5.0):
        self.agent_id = agent_id
        self.spread_pct = spread
        
    def observe_return(self, *args):
        pass # Value investors don't care about price history
        
    def generate_bid_ask_spread(self, current_price, current_volatility, current_time, true_value, **kwargs):
        dynamic_spread = self.spread_pct * current_volatility 
        return {
            'agent_id': self.agent_id,
            'bid': true_value * (1 - (0.5 * dynamic_spread)),
            'ask': true_value * (1 + (0.5 * dynamic_spread))
        }