class ValueAgent:
    def __init__(self, agent_id, spread=5.0):
        self.agent_id = agent_id
        self.spread_pct = spread
        self.cash = 10000.0
        self.shares = 300
        
    def observe_return(self, *args):
        pass # Value investors don't care about price history
        
    def generate_bid_ask_spread(self, current_price, current_volatility, current_time, true_value, **kwargs):
        dynamic_spread = self.spread_pct * current_volatility
        spend_ratio = 0.10 # use 10% of available capital/shares per order
        orders = {'agent_id': self.agent_id}
        
        if true_value >= current_price: # directional trading: buy if agent thinks asset is underpriced, sell if overpriced
            bid_price = true_value * (1 - (0.5 * dynamic_spread))
            volume = int((self.cash * spend_ratio) / bid_price)
            if volume > 0:
                orders['bid'] = bid_price
                orders['bid_volume'] = volume
                
        elif true_value < current_price:
            ask_price = true_value * (1 + (0.5 * dynamic_spread))
            volume = int(self.shares * spend_ratio)
            if volume > 0:
                orders['ask'] = ask_price
                orders['ask_volume'] = volume

        return orders if len(orders) > 1 else None