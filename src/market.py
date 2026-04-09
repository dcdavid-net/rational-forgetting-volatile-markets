# The OrderBook and Matching Engine

class Order:
    def __init__(self, order_type, agent_id, price, volume, timestamp):
        self.order_type = order_type  # 'bid' or 'ask'
        self.agent_id = agent_id
        self.price = price
        self.volume = volume
        self.timestamp = timestamp
    
    def __repr__(self):
        return f"Order({self.order_type}, price={self.price}, vol={self.volume}, agent={self.agent_id}, t={self.timestamp})"

class Market:
    def __init__(self):
        self.bids = []  # Buy orders sorted highest price, oldest
        self.asks = []  # Sell orders sorted lowest price, oldest
        self.trade_history = []  # Record of all executed trade prices
        self.clock = 0  # Internal clock to enforce time priority

    def submit_order(self, order_type, agent_id, price, volume):
        self.clock += 1
        new_order = Order(order_type, agent_id, price, volume, self.clock)
        return self._match(new_order)

    def cancel_agent_orders(self, agent_id):
        self.bids = [order for order in self.bids if order.agent_id != agent_id]
        self.asks = [order for order in self.asks if order.agent_id != agent_id]

    def _match(self, incoming_order):
        if incoming_order.order_type not in ('ask','bid'):
            return []
        
        is_bid = incoming_order.order_type == 'bid'
        if is_bid:
            orders = self.bids
            opposing_orders = self.asks
            price_sort = -1
        else:
            orders = self.asks
            opposing_orders = self.bids
            price_sort = 1

        executed_trades = []

        # as long as there is an overlapping price AND the incoming order still has volume
        # lets try to keep matching
        while opposing_orders and incoming_order.volume > 0:
            best_opposing = opposing_orders[0]

            # if no price overlap, stop
            if is_bid and incoming_order.price < best_opposing.price:
                break
            if not is_bid and incoming_order.price > best_opposing.price:
                break

            trade_price = best_opposing.price
            trade_volume = min(incoming_order.volume, best_opposing.volume)

            executed_trades.append({
                'buyer_id': incoming_order.agent_id if is_bid else best_opposing.agent_id,
                'seller_id': best_opposing.agent_id if is_bid else incoming_order.agent_id,
                'price': trade_price,
                'volume': trade_volume,
                'timestamp': self.clock
            }) 
            self.trade_history.append(trade_price)
            
            # deduct volume from both sides after resolving order
            incoming_order.volume -= trade_volume
            best_opposing.volume -= trade_volume

            # if resting order was completely wiped, let's remove it
            if best_opposing.volume == 0:
                opposing_orders.pop(0)

        # if incoming order still has volume, record it
        if incoming_order.volume > 0:
            orders.append(incoming_order)
            orders.sort(key=lambda x: (x.price * price_sort, x.timestamp))

        return executed_trades
    
    def get_latest_price(self):
        if self.trade_history:
            return self.trade_history[-1]
        return None