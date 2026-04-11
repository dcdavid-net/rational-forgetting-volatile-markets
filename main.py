# The main runner of the whole project

import csv
from math import log, sqrt
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import kurtosis
from src.agent import Agent
from src.value_agent import ValueAgent
from src.market import Market
from src.generator import generate_fundamental_value

# Set global seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# For Phase 2: monte carlo. May later need to set different seeds per run
def set_reproducibility_seed(run_id=0):
    seed_value = 42 + run_id
    random.seed(seed_value)
    np.random.seed(seed_value)

def get_log_return(current_price, previous_price):
    return log(current_price / previous_price) if previous_price > 0 else 0.0

def get_EWMA_variance(ewma_lambda, current_variance, r_t_minus_1):
    return max(ewma_lambda * current_variance + (1 - ewma_lambda) * (r_t_minus_1 ** 2), 1e-8)

def get_volatility(current_variance):
    return sqrt(current_variance)

def init(num_agents, decay_rate, start_price, prune_threshold):
    market = Market()
    
    # mix of IBL and Value agents. Aiming for 10-20% Value
    num_value = int(num_agents * 0.15)
    num_ibl = num_agents - num_value
    agents = [Agent(agent_id=i, decay_rate=decay_rate, spread=0.5, prune_threshold=prune_threshold) for i in range(num_ibl)]
    agents += [ValueAgent(agent_id=i+num_ibl, spread=0.5) for i in range(num_value)]
    
    # BOTH current and previous price must initialize to the exact start price
    current_price = start_price
    previous_price = start_price
    ewma_lambda = 0.94
    current_variance = 1.0  # \sigma_0^2
    current_volatility = sqrt(current_variance)
    price_history = []
    return market, agents, current_price, previous_price, ewma_lambda, current_variance, current_volatility, price_history

def run_simulation(num_agents=50, total_steps=5000, burn_in=5000, decay_rate=0.5, start_price=100, prune_threshold=-2.292483739335286, volatility_scaling=0.01):
    set_reproducibility_seed()
    
    V_t = generate_fundamental_value(steps=total_steps + burn_in + 1, start_price=start_price, volatility=volatility_scaling) # + 1 because 1-based indexing loop
    market, agents, current_price, previous_price, ewma_lambda, current_variance, current_volatility, price_history = \
        init(num_agents, decay_rate, start_price=V_t[0], prune_threshold=prune_threshold)
    
    print(f'Starting simulation: N={num_agents}, d={decay_rate}, steps={total_steps}, burn-in={burn_in}')
    agent_dict = {a.agent_id: a for a in agents}
    for t in range(1, total_steps + burn_in + 1):
        # PHASE 1: BURN-IN (EXOGENOUS) to observe the random-walk market and build some memory
        if t <= burn_in:
            previous_price = current_price
            current_price = V_t[t]  # Exogenous price discovery dictates the market
            r_t_minus_1 = get_log_return(current_price, previous_price)
            current_variance = get_EWMA_variance(ewma_lambda, current_variance, r_t_minus_1)
            current_volatility = get_volatility(current_variance)
            for agent in agents:
                agent.observe_return(r_t_minus_1, current_volatility, t)
                
        # PHASE 2: CONTINUOUS DOUBLE AUCTION (Endogenous Price)
        else:
            r_t_minus_1 = get_log_return(current_price, previous_price)
            current_variance = get_EWMA_variance(ewma_lambda, current_variance, r_t_minus_1)
            current_volatility = get_volatility(current_variance)
            random.shuffle(agents) # prevent structural ordering bias in the Continuous Double Auction
            
            # give the agents some salary like in the real world
            # this keeps everyone participating especially since
            # IBL agents are at disadvantage vs Value Agents.
            # This keeps IBL agents in the market when wealth transfer happens
            if (t - burn_in) % 30 == 0:
                salary_amount = 50.0 # 0.5% of starting cash
                for agent in agents:
                    agent.cash += salary_amount

            # IBL Agents now actively trade based on their populated memory
            # Value Agents should just trade based on fundamentals
            for agent in agents:
                agent.observe_return(r_t_minus_1, current_volatility, t)
                market.cancel_agent_orders(agent.agent_id) # cancel old orders to conserve computer memory
                orders = agent.generate_bid_ask_spread(
                    current_price, 
                    current_volatility, 
                    t, 
                    true_value=V_t[t], 
                    do_pruning=True
                )
                
                if orders:
                    if 'bid' in orders and orders.get('bid_volume', 0) > 0:
                        executed_trades = market.submit_order('bid', agent.agent_id, orders['bid'], orders['bid_volume'])
                    if 'ask' in orders and orders.get('ask_volume', 0) > 0:
                        executed_trades = market.submit_order('ask', agent.agent_id, orders['ask'], orders['ask_volume'])
                        
                    for trade in executed_trades:
                        buyer = agent_dict[trade['buyer_id']]
                        seller = agent_dict[trade['seller_id']]
                        
                        buyer.cash -= trade['price'] * trade['volume']
                        buyer.shares += trade['volume']
                        seller.cash += trade['price'] * trade['volume']
                        seller.shares -= trade['volume']
                    
            previous_price = current_price
            latest_trade_price = market.get_latest_price() # Price updates strictly based on CDA execution, completely blind to V_t
            
            if latest_trade_price is not None:
                current_price = max(latest_trade_price, 1.0)
                
            price_history.append(current_price)

        if t % 500 == 0:
            print(f'Step {t}: Price = {current_price:.2f}, Volatility = {current_volatility:.4f}')
            ibl_cash = sum(a.cash for a in agents if isinstance(a, Agent))
            val_cash = sum(a.cash for a in agents if isinstance(a, ValueAgent))
            ibl_shares = sum(a.shares for a in agents if isinstance(a, Agent))
            val_shares = sum(a.shares for a in agents if isinstance(a, ValueAgent))
            print(f"IBL Total Cash: ${ibl_cash:.2f} | Shares: {ibl_shares} | Net Worth: {ibl_cash + (ibl_shares*current_price)}")
            print(f"VAL Total Cash: ${val_cash:.2f} | Shares: {val_shares} | Net Worth: {val_cash + (val_shares*current_price)}")

            kurt, acar = calculate_metrics(price_history)
            print(f"Running Kurtosis and ACAR: {kurt:.4f} {acar:.4f}")

    return price_history, V_t[burn_in + 1:]

def calculate_metrics(prices):
    prices_array = np.array(prices)

    # kurtosis
    returns = np.log(prices_array[1:] / prices_array[:-1])
    kurt = kurtosis(returns, fisher=True)
    
    # autocorrelation of absolute returns (Volatility Clustering)
    abs_returns = np.abs(returns)
    if len(abs_returns) > 1:
        acar = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
    else:
        acar = 0.0
    return kurt, acar

if __name__ == "__main__":
    # Baseline A: Zero-Decay Control (Rational EMH)
    print('Running Baseline A (d=0.0)...')
    baseline_prices, V_t_array = run_simulation(decay_rate=0.0, start_price=100, volatility_scaling=0.01)
    
    # Treatment A: Human Decay Model
    print('\nRunning Treatment A (d=0.5, tau=-2.29)...')
    treatment_a_prices, _ = run_simulation(decay_rate=0.5, start_price=100, volatility_scaling=0.01)

    # print('Running Treatment B (d=0.5, tau=-10.0)...')
    # treatment_b_prices, _ = run_simulation(decay_rate=0.5, start_price=100, prune_threshold=1.0, volatility_scaling=0.01)

    with open('data/results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['fundamental_value', 'baseline_price', 'treatment_a_price']) #, 'treatment_b_price'])
        writer.writerows(zip(V_t_array, baseline_prices, treatment_a_prices)) #, treatment_b_prices))
    
    print(f'Saved to data/results.csv')

    fundamental_kurt, fundamental_acar = calculate_metrics(V_t_array)
    base_kurt, base_acar = calculate_metrics(baseline_prices)
    treat_a_kurt, treat_a_acar = calculate_metrics(treatment_a_prices)
    # treat_b_kurt, treat_b_acar = calculate_metrics(treatment_b_prices)
    
    print('FINAL MARKET METRICS')
    print(f'Fundamental Value:')
    print(f'  Fisher Kurtosis : {fundamental_kurt:.4f} (Expected ~0.0 for Random Walk)')
    print(f'  Abs Returns AC : {fundamental_acar:.4f} (Expected low for Random Walk)')
    print(f'Baseline A (d=0.0):')
    print(f'  Fisher Kurtosis : {base_kurt:.4f} (Expected ~0.0 for Random Walk)')
    print(f'  Abs Returns AC : {base_acar:.4f} (Expected low for Random Walk)')
    print(f'\nTreatment A (d=0.5, tau=-2.29):')
    print(f'  Fisher Kurtosis : {treat_a_kurt:.4f} (>0 indicates Fat Tails/Crashes)')
    print(f'  Abs Returns AC : {treat_a_acar:.4f} (>0 indicates Volatility Clustering)')
    # print(f'\nTreatment B (d=0.5, tau=1.0):')
    # print(f'  Fisher Kurtosis : {treat_b_kurt:.4f} (>0 indicates Fat Tails/Crashes)')
    # print(f'  Abs Returns AC : {treat_b_acar:.4f} (>0 indicates Volatility Clustering)')
    
    print('Generate chart')
    GT_GOLD = "#B3A369"
    GT_NAVY = "#003057"
    GT_GRAY = "#545454"
    plt.figure(figsize=(12, 6))
    plt.plot(V_t_array, label="Fundamental Value (V_t)", color=GT_GRAY, linestyle='dashed', linewidth=1.5)
    plt.plot(baseline_prices, label="Baseline A (d=0.0)", color=GT_NAVY, linewidth=1)
    plt.plot(treatment_a_prices, label="Treatment A (d=0.5)", color=GT_GOLD, linewidth=1)
    # plt.plot(treatment_b_prices, label="Treatment B (d=0.5, tau=1.0)", color=GT_NAVY, linewidth=1)
    plt.title('Simulation Output: Endogenous Price Paths')
    plt.xlabel('Time Step (Post Burn-In)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Baseline vs Treatment.png')
    plt.show()