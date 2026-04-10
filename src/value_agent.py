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
        spend_ratio = 0.05 # use 5% of available capital/shares per order
        orders = {'agent_id': self.agent_id}

        # Cash + Value of holdings to determine shorting ability / margin for the account
        equity = self.cash + (self.shares * current_price)
        if equity <= 0: # agent is bankrupt, they cannot trade
            return None

        market_price = max(current_price, 0.01) # tick size of the market. Price can never truly be 0.0
        if true_value >= current_price: # directional trading: buy if agent thinks asset is underpriced, sell if overpriced
            # agents in the stock market put bids/asks based on what the market price is not necessarily what they think is fair. 
            # Example: stock price = 100. If agent thinks the stock price should be at 200, they dont just place a bid at 200.
            #          Instead, they should bid at 100 (market price) so that they have money to gain on the way up.
            bid_price = current_price * (1 + (0.5 * dynamic_spread))
            target_volume = int((self.cash * spend_ratio) / market_price) # size the order based on market price
            max_affordable = int(self.cash / bid_price) if bid_price > 0 else 0 # ensure agent can actually afford its bid/ask
            volume = min(target_volume, max_affordable)
            
            if volume > 0:
                orders['bid'] = bid_price
                orders['bid_volume'] = volume
                
        elif true_value < current_price:
            ask_price = current_price * (1 - (0.5 * dynamic_spread))
            
            # FINRA Rule 4210
            # low-priced stocks is $2.50 per share on margin
            margin_per_share = max(market_price, 2.50)
            target_volume = int((equity * spend_ratio) / margin_per_share)
            max_short = int(equity / margin_per_share)
            volume = min(target_volume, max_short)
            
            if volume > 0:
                orders['ask'] = ask_price
                orders['ask_volume'] = volume

        return orders if len(orders) > 1 else None