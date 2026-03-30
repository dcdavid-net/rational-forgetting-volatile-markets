# Developer Notes for working with this Repo

Directory Tree (including Local Files)
rational-forgetting-volatile-markets/
├── academic_sources/
│   ├── ACT-R Manual.pdf
│   ├── Anderson -- Integrated Theory of Mind.pdf
│   ├── Anderson -- Is human cognition adaptive.pdf
│   ├── Anderson -- The Atomic Components of Thought.pdf
│   ├── Gonzalez -- Instance-based Learning in Dynamic Decision Making.pdf
│   ├── Jiang -- Investor Memory and Biased Beliefs Evidence from the Field.pdf
│   ├── Lieder -- Resource-rational analysis Understanding human cognition as the optimal use of.pdf
│   ├── Malmendier -- Depression Babies Do Macroeconomic Experiences Affect Risk-Taking.pdf
│   ├── RiskMetrics Technical Document.pdf
│   └── Welch -- A Heuristic for Fat-Tailed Stock Market Returns.pdf
├── data/
│   └── sp500_historical_returns.csv
├── notebooks/ 
├── src/
│   ├── agent.py
│   ├── calibrate_pruning.py
│   ├── fetch_sp500.py
│   ├── generator.py
│   ├── market.py
│   └── tester.py
├── .gitignore
├── main.py
├── outputs.txt
├── planning.csv
├── README.md
├── requirements.txt
├── THEORY.md
└── TODO.md

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