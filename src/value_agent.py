class ValueAgent:
    def __init__(self, agent_id, spread=5.0):
        self.agent_id = agent_id
        self.spread_pct = spread
        self.cash = 10000.0
        self.shares = 100
        
    def observe_return(self, *args):
        pass # Value investors don't care about price history
        
    def generate_bid_ask_spread(self, current_price, current_volatility, current_time, true_value, **kwargs):
        dynamic_spread = self.spread_pct * current_volatility
        orders = {'agent_id': self.agent_id}
        bid_price = expected_value * (1 - (0.5 * dynamic_spread))
        orders['bid'] = bid_price if self.cash >= bid_price else None
        orders['ask'] = expected_value * (1 + (0.5 * dynamic_spread)) if self.shares > 0 else None

        return orders if len(orders) > 1 else None