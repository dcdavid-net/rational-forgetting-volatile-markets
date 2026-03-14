# The main runner of the whole project

import random
import numpy as np
import matplotlib.pyplot as plt
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