# The main runner of the whole project

import csv
from math import log, sqrt
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import kurtosis
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import mannwhitneyu # non parametric statistical significance
import warnings

from src.agent import Agent
from src.value_agent import ValueAgent
from src.market import Market
from src.generator import generate_fundamental_value

# Set global seeds for reproducibility
RANDOM_SEED = 42

def set_reproducibility_seed(run_id=0):
    seed_value = RANDOM_SEED + run_id
    random.seed(seed_value)
    np.random.seed(seed_value)

def get_log_return(current_price, previous_price):
    return log(current_price / previous_price) if previous_price > 0 else 0.0

def get_EWMA_variance(ewma_lambda, current_variance, r_t_minus_1):
    return max(ewma_lambda * current_variance + (1 - ewma_lambda) * (r_t_minus_1 ** 2), 1e-8)

def get_volatility(current_variance):
    return sqrt(current_variance)

def init(num_agents, decay_rate, start_price, prune_threshold, value_ratio):
    market = Market()
    
    # mix of IBL and Value agents. Aiming for 10-20% Value
    num_value = int(num_agents * value_ratio)
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

def run_simulation(num_agents=50, total_steps=5000, burn_in=5000, decay_rate=0.5, start_price=100, prune_threshold=-2.292483739335286, volatility_scaling=0.01, run_id=0, value_ratio=0.15, verbose=False):
    set_reproducibility_seed(run_id)
    
    V_t = generate_fundamental_value(steps=total_steps + burn_in + 1, start_price=start_price, volatility=volatility_scaling) # + 1 because 1-based indexing loop
    market, agents, current_price, previous_price, ewma_lambda, current_variance, current_volatility, price_history = \
        init(num_agents, decay_rate, start_price=V_t[0], prune_threshold=prune_threshold, value_ratio=value_ratio)
    
    if verbose:
        print(f'Starting simulation (Run ID: {run_id}): N={num_agents}, d={decay_rate}, Value Ratio={value_ratio}')
        
    agent_dict = {a.agent_id: a for a in agents}
    activations_log = {} # metric #2  Activation States --- to explain bubble
    
    for t in range(1, total_steps + burn_in + 1):
        # PHASE 1: BURN-IN (EXOGENOUS) to observe the random-walk market and build some memory
        if t <= burn_in:
            previous_price = current_price
            current_price = V_t[t]
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
            
            if (t - burn_in) % 30 == 0:
                salary_amount = 50.0 # 0.5% of starting cash
                for agent in agents:
                    agent.cash += salary_amount

            for agent in agents:
                agent.observe_return(r_t_minus_1, current_volatility, t)
                
                # Metric # 5 Activation States --- to explain bubble
                if run_id == 0 and getattr(agent, 'agent_id', -1) == 0 and hasattr(agent, 'memory'):
                    mem_1 = agent.memory.get(1, [])
                    mem_5 = agent.memory.get(5, [])
                    
                    ts_1 = [ts for ts, _ in mem_1]
                    ts_5 = [ts for ts, _ in mem_5]
                    
                    b1 = agent._get_base_level_activation(ts_1, t) if ts_1 else float('-inf')
                    s1 = agent._get_contextual_similarity(mem_1, current_volatility) if mem_1 else float('-inf')
                    
                    b5 = agent._get_base_level_activation(ts_5, t) if ts_5 else float('-inf')
                    s5 = agent._get_contextual_similarity(mem_5, current_volatility) if mem_5 else float('-inf')
                    
                    activations_log[t] = {'b1': b1, 's1': s1, 'b5': b5, 's5': s5, 'vol': current_volatility}

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
            latest_trade_price = market.get_latest_price() # Price updates strictly based on CDA execution, completely blind to V_t aside from Value Agents
            
            if latest_trade_price is not None:
                current_price = max(latest_trade_price, 1.0)
                
            price_history.append(current_price)

        if verbose and t % 500 == 0:
            print(f'[Run {run_id}] Step {t}: Price = {current_price:.2f}, Volatility = {current_volatility:.4f}')

    # Limits to Arbitrage -- do value agents get insolvent as market stays irrational?
    ibl_net_worth = sum(a.cash + (a.shares * current_price) for a in agents if isinstance(a, Agent))
    val_net_worth = sum(a.cash + (a.shares * current_price) for a in agents if isinstance(a, ValueAgent))

    return price_history, V_t[burn_in + 1:], ibl_net_worth, val_net_worth, activations_log

def calculate_metrics(prices):
    prices_array = np.array(prices)
    returns = np.log(prices_array[1:] / prices_array[:-1])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kurt = kurtosis(returns, fisher=True)
        
    abs_returns = np.abs(returns)
    if len(abs_returns) > 1:
        acar = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
    else:
        acar = 0.0
    return kurt, acar

def monte_carlo_worker(run_id):
    # 1. Structural Benchmark: 100% Value Agents (Isolates microstructure friction)
    prices_val, vt_val, ibl_nw_val, val_nw_val, _ = run_simulation(run_id=run_id, value_ratio=1.0, decay_rate=0.0, verbose=False)
    kurt_val, acar_val = calculate_metrics(prices_val)
    kurt_vt, acar_vt = calculate_metrics(vt_val) 
    
    # 2. Behavioral Control: 90% IBL (d=0.0) / 10% Value
    prices_base, _, ibl_nw_base, val_nw_base, _ = run_simulation(run_id=run_id, value_ratio=0.10, decay_rate=0.0, verbose=False)
    kurt_base, acar_base = calculate_metrics(prices_base)
    
    # 3. Experimental Treatment: 90% IBL (d=0.5) / 10% Value
    prices_treat, _, ibl_nw_treat, val_nw_treat, act_log_treat = run_simulation(run_id=run_id, value_ratio=0.10, decay_rate=0.5, verbose=False)
    kurt_treat, acar_treat = calculate_metrics(prices_treat)
    
    # Metric #3 and #5 are kurtosis and ACAR
    # Metric #1 Bubble Indicator / maximum deviation
    max_dev_base = np.max(np.array(prices_base) / np.array(vt_val))
    max_dev_treat = np.max(np.array(prices_treat) / np.array(vt_val))

    result = {
        'run_id': run_id,
        'kurt_vt': kurt_vt, 'acar_vt': acar_vt,
        'kurt_val': kurt_val, 'acar_val': acar_val,
        'kurt_base': kurt_base, 'acar_base': acar_base,
        'kurt_treat': kurt_treat, 'acar_treat': acar_treat,
        'max_dev_base': max_dev_base,
        'max_dev_treat': max_dev_treat,
        'ibl_nw_treat': ibl_nw_treat,
        'val_nw_treat': val_nw_treat
    }
    
    if run_id == 0:
        result['paths'] = {
            'vt': vt_val,
            'val': prices_val,
            'base': prices_base,
            'treat': prices_treat,
            'activations': act_log_treat
        }
        
    return result

if __name__ == "__main__":
    NUM_RUNS = 100
    MAX_WORKERS = 18 
    
    print(f"Starting Monte Carlo Simulation...")
    print(f"Iterations: {NUM_RUNS}")
    print(f"Parallel Workers: {MAX_WORKERS}\n")
    
    results = []
    paths_for_chart = None
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(monte_carlo_worker, i): i for i in range(NUM_RUNS)}
        
        for count, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                results.append(res)
                print(f"[{count}/{NUM_RUNS}] Completed Run ID: {res['run_id']}")
                if 'paths' in res:
                    paths_for_chart = res['paths']
            except Exception as e:
                print(f"Run generated an exception: {e}")

    # Aggregate Statistics
    metrics = {
        'Fundamental': {'kurt': [], 'acar': []},
        'Val_Benchmark': {'kurt': [], 'acar': []},
        'Behavioral_Control': {'kurt': [], 'acar': []},
        'Treatment': {'kurt': [], 'acar': []}
    }
    
    max_dev_base_list = []
    max_dev_treat_list = []
    ibl_nw_treat_list = []
    val_nw_treat_list = []
    
    for r in results:
        metrics['Fundamental']['kurt'].append(r['kurt_vt'])
        metrics['Fundamental']['acar'].append(r['acar_vt'])
        metrics['Val_Benchmark']['kurt'].append(r['kurt_val'])
        metrics['Val_Benchmark']['acar'].append(r['acar_val'])
        metrics['Behavioral_Control']['kurt'].append(r['kurt_base'])
        metrics['Behavioral_Control']['acar'].append(r['acar_base'])
        metrics['Treatment']['kurt'].append(r['kurt_treat'])
        metrics['Treatment']['acar'].append(r['acar_treat'])
        
        max_dev_base_list.append(r['max_dev_base'])
        max_dev_treat_list.append(r['max_dev_treat'])
        ibl_nw_treat_list.append(r['ibl_nw_treat']) # Metric #4, limits to arbitrage. Markets remain irrational longer than agent can be solvent
        val_nw_treat_list.append(r['val_nw_treat'])

    print('########### FINAL MONTE CARLO METRICS (N=100 Runs) ############')
    print("95% Confidence Interval")
    
    for name, data in metrics.items():
        kurt_mean = np.nanmean(data['kurt'])
        kurt_ci = np.nanpercentile(data['kurt'], [2.5, 97.5])
        acar_mean = np.nanmean(data['acar'])
        acar_ci = np.nanpercentile(data['acar'], [2.5, 97.5])
        
        print(f"{name}:")
        print(f"  Fisher Kurtosis : {kurt_mean:.4f}  [{kurt_ci[0]:.4f}, {kurt_ci[1]:.4f}]")
        print(f"  Abs Returns AC  : {acar_mean:.4f}  [{acar_ci[0]:.4f}, {acar_ci[1]:.4f}]\n")

    print('---------------------------------------------------------------')
    print("METRIC 4: LIMITS TO ARBITRAGE (WEALTH DISTRIBUTION)")
    print(f"Average Final IBL Net Worth (Treatment) : ${np.mean(ibl_nw_treat_list):,.2f}")
    print(f"Average Final VAL Net Worth (Treatment) : ${np.mean(val_nw_treat_list):,.2f}")
    
    print('---------------------------------------------------------------')
    print("METRIC 1: MAX DEVIATION FROM FUNDAMENTAL (BUBBLE SIZE)")
    print(f"Behavioral Control (d=0.0) Avg Max Dev : {np.mean(max_dev_base_list):.2f}x")
    print(f"Experimental Treatment (d=0.5) Avg Max Dev: {np.mean(max_dev_treat_list):.2f}x")
    print('---------------------------------------------------------------')

    # Mann-Whitney U -- statistical significance for non parametric
    print("HYPOTHESIS TESTING: DOES BIOLOGICAL DECAY CAUSE CRASHES?")

    # Extract the arrays of 100 kurtosis values, filtering out any NaNs just in case
    control_kurtosis = [k for k in metrics['Behavioral_Control']['kurt'] if not np.isnan(k)]
    treatment_kurtosis = [k for k in metrics['Treatment']['kurt'] if not np.isnan(k)]

    u_stat, p_value = mannwhitneyu(treatment_kurtosis, control_kurtosis, alternative='greater')

    print(f"Mann-Whitney U Statistic : {u_stat}")
    print(f"P-Value                  : {p_value:.6f}")

    if p_value < 0.05: # standard alpha = 0.05
        print("\nCONCLUSION: STATISTICALLY SIGNIFICANT (p < 0.05)")
        print("Reject the null hypothesis. The introduction of biological memory decay (d=0.5)")
        print("causes a statistically significant increase in market fat-tails (crashes)")
        print("compared to the perfectly rational baseline (d=0.0).")
    else:
        print("\nCONCLUSION: NOT STATISTICALLY SIGNIFICANT (p >= 0.05)")
        print("Fail to reject the null hypothesis. There is not enough evidence to prove")
        print("that the decay rate caused a statistically significant increase in market fat-tails.")
        
    print('---------------------------------------------------------------')
    print("HYPOTHESIS TESTING: DOES BIOLOGICAL DECAY CAUSE VOLATILITY CLUSTERING?")

    control_acar = [a for a in metrics['Behavioral_Control']['acar'] if not np.isnan(a)]
    treatment_acar = [a for a in metrics['Treatment']['acar'] if not np.isnan(a)]

    u_stat_acar, p_value_acar = mannwhitneyu(treatment_acar, control_acar, alternative='greater')

    print(f"Mann-Whitney U Statistic : {u_stat_acar}")
    print(f"P-Value                  : {p_value_acar:.6f}")

    if p_value_acar < 0.05:
        print("\nCONCLUSION: STATISTICALLY SIGNIFICANT (p < 0.05)")
        print("Reject the null hypothesis. The biological memory decay (d=0.5)")
        print("causes a statistically significant increase in volatility clustering (ACAR)")
        print("compared to the perfectly rational baseline.")
    else:
        print("\nCONCLUSION: NOT STATISTICALLY SIGNIFICANT (p >= 0.05)")
        print("Fail to reject the null hypothesis. There is not enough evidence to prove")
        print("that the decay rate caused an increase in volatility clustering.")

    print('---------------------------------------------------------------')

    if paths_for_chart:
        print('Saving representative paths (Run ID 0) to data/main_run_0_results.csv...')
        with open('data/main_run_0_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['fundamental_value', 'benchmark_val', 'baseline_price', 'treatment_price'])
            writer.writerows(zip(paths_for_chart['vt'], paths_for_chart['val'], paths_for_chart['base'], paths_for_chart['treat']))
        
        # Metric #2 Activation States --- to explain bubble
        # Find the peak bubble time step in the representative treatment run
        treat_prices_arr = np.array(paths_for_chart['treat'])
        vt_prices_arr = np.array(paths_for_chart['vt'])
        deviations = treat_prices_arr / vt_prices_arr
        peak_idx = np.argmax(deviations)
        
        # Adjust for the 5000-step burn-in offset (since prices_arr starts at t=5001)
        peak_t = 5000 + 1 + peak_idx 
        
        act_log = paths_for_chart['activations']
        if peak_t in act_log:
            peak_data = act_log[peak_t]
            print("\nMETRIC 2: Activation States --- to explain bubble")
            print(f"At peak bubble (t={peak_t}), Agent 0's internal memory state:")
            print(f"Current Volatility: {peak_data['vol']:.6f}")
            print(f"Bin 5 (Extreme Positive) - Base Act (B_i): {peak_data['b5']:.2f} | Context Penalty (S_ctx): {peak_data['s5']:.2f} | Total Expected: {peak_data['b5']+peak_data['s5']:.2f}")
            print(f"Bin 1 (Extreme Negative) - Base Act (B_i): {peak_data['b1']:.2f} | Context Penalty (S_ctx): {peak_data['s1']:.2f} | Total Expected: {peak_data['b1']+peak_data['s1']:.2f}")
            print("===========================================\n")

        print('Generating charts...')
        GT_GOLD = "#B3A369"
        GT_NAVY = "#003057"
        GT_GRAY = "#545454"
        GT_LBLUE = "#2961FF"
        
        # Original Chart
        plt.figure(figsize=(14, 7))
        plt.plot(paths_for_chart['vt'], label="Fundamental Value (V_t)", color=GT_GRAY, linestyle='dashed', linewidth=2.0)
        plt.plot(paths_for_chart['val'], label="100% Value Agents", color=GT_LBLUE, linewidth=1.25, alpha=0.5)
        plt.plot(paths_for_chart['base'], label="Behavioral Control (d=0.0)", color=GT_NAVY, linewidth=0.75)
        plt.plot(paths_for_chart['treat'], label="Experimental Treatment (d=0.5)", color=GT_GOLD, linewidth=0.75)
        plt.title('Simulation Output: Endogenous Price Paths (Representative Instance)')
        plt.xlabel('Time Step (Post Burn-In)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('Baseline vs Treatment.png')
        # plt.show()

        # Metric #3 Fat Tails histogram
        returns_treat = np.diff(np.log(paths_for_chart['treat']))
        plt.figure(figsize=(10, 6))
        plt.hist(returns_treat, bins=50, density=True, alpha=0.6, color=GT_GOLD, label="Treatment Log-Returns")
        
        mu, std = np.mean(returns_treat), np.std(returns_treat)
        x = np.linspace(min(returns_treat), max(returns_treat), 100)
        p = np.exp(-0.5 * ((x - mu) / std)**2) / (std * np.sqrt(2 * np.pi))
        plt.plot(x, p, 'k', linewidth=2, label="Normal Distribution (Theoretical)")
        
        plt.title('Metric 3: Visual Proof of Fat Tails (Return Distribution)')
        plt.xlabel('Log Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('Fat_Tails_Histogram.png')
        plt.show()

        # Metric #5 Volatility Clustering ACAR
        plt.figure(figsize=(14, 4))
        plt.bar(range(len(returns_treat)), np.abs(returns_treat), color=GT_NAVY, width=1.0)
        plt.title('Metric 5: Visual Proof of Volatility Clustering (Absolute Returns)')
        plt.xlabel('Time Step')
        plt.ylabel('Absolute Log Return')
        plt.grid(True, alpha=0.3)
        plt.savefig('Volatility_Clustering.png')
        plt.show()