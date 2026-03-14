# Agent Class

class Agent:
    def __init__(self, agent_id, decay_rate=0.5, prune_threshold=-10.0, spread=2.0):
        self.agent_id = agent_id
        self.decay_rate = decay_rate
        self.prune_threshold = prune_threshold
        self.spread = spread # static for now. Later in Phase 2, may have to make this variable

        self.memory = {}
    
    def __repr__(self):
        return f"Agent({self.agent_id}, decay_rate={self.decay_rate}, prune_threshold={self.prune_threshold}, spread={self.spread})"

    def observe_price(self, price, current_time):
        if price not in self.memory:
            self.memory[price] = []
        self.memory[price].append(current_time)