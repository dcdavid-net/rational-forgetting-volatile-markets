# Developer Notes for working with this Repo

- src/agent.py is the IBL Agent class

- src/calibrate_pruning.py generates -2.29 as the pruning threshold for my report

- src/fetch_sp500.py generates
  - sp500_historical_returns.csv

- src/generator.py generates a Geometric random walk for main.py

- market.py is the Market class

- tester.py tests most of my scripts and generates outputs.txt

- src/tester.py generates
  - outputs.txt

- src/value_agent.py is the Value Agent Class

- main.py generates
  - Baseline vs Treatment.png
  - Fat_Tails_Histogram.png
  - Volatility_Clustering.png
  - main_output.txt (through the command line, and those results are copy pasted to this txt file)
  - data/main_run_0_results.csv

Initial Setup
```
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

Run test script
```
python -m src.tester
```