# The OrderBook and Matching Engine

class Order:
    def __init__(self, order_type, agent_id, price, timestamp):
        self.order_type = order_type  # 'bid' or 'ask'
        self.agent_id = agent_id
        self.price = price
        self.timestamp = timestamp
    
    def __repr__(self):
        return f"Order({self.order_type}, price={self.price}, agent={self.agent_id}, t={self.timestamp})"

class Market:
    def __init__(self):
        self.bids = []  # Buy orders sorted highest price, oldest
        self.asks = []  # Sell orders sorted lowest price, oldest
        self.trade_history = []  # Record of all executed trade prices
        self.clock = 0  # Internal clock to enforce time priority

    def submit_order(self, order_type, agent_id, price):
        self.clock += 1
        new_order = Order(order_type, agent_id, price, self.clock)
        return self._match(new_order)

    def _match(self, incoming_order):
        if incoming_order.order_type not in ('ask','bid'):
            return []
        
        is_bid = incoming_order.order_type == 'bid'
        if is_bid:
            has_matching_order = self.asks and incoming_order.price >= self.asks[0].price
            orders = self.bids
            opposing_orders = self.asks
            price_sort = -1
        else:
            has_matching_order = self.bids and incoming_order.price <= self.bids[0].price
            orders = self.asks
            opposing_orders = self.bids
            price_sort = 1

        if has_matching_order:
            executed_trades = []
            best_opposing_order = opposing_orders.pop(0)
            trade_or_resting_price = best_opposing_order.price
            executed_trades.append({
                'buyer_id': incoming_order.agent_id if is_bid else best_opposing_order.agent_id,
                'seller_id': best_opposing_order.agent_id if is_bid else incoming_order.agent_id,
                'price': trade_or_resting_price,
                'timestamp': self.clock
            }) # TODO: Later on, there may be a possibility of agents putting in multiple orders.
            self.trade_history.append(trade_or_resting_price)
            return executed_trades
        else:
            orders.append(incoming_order)
            orders.sort(key=lambda x: (x.price * price_sort, x.timestamp))
            return []
    
    def get_latest_price(self):
        if self.trade_history:
            return self.trade_history[-1]
        return None