# Author: Mario Niemann

import json
import numpy as np
import torch
import random
import os
from datetime import datetime

from modules.experiment import run_experiment
from modules.analysis import analyze_results

# Function to set the selected random seed in all relevant places
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# the main experiment loop
def experiment_loop(resultfolder, algorithms, environments, random_seeds):
    # TODO: set random seed randomly in case none was provided
    if random_seeds:
        for seed in random_seeds:
            set_random_seed(seed)
            for algorithm in algorithms:
                for environment in environments:
                    run_experiment({"resultfolder":resultfolder, "environment": environment}, algorithm, environments[environment], seed)



if __name__=='__main__':
    # open and read config file
    with open("./config.json","r") as jsonfile:
        data = json.load(jsonfile)

    # define and create results folder
    current = datetime.now()
    resultfolder = f"./results/{current.isoformat()}_{data.get('experiment_name')}"
    if not os.path.exists(resultfolder):
        os.makedirs(resultfolder)

    # run the experiment
    experiment_loop(resultfolder,data.get('algorithms'), data.get('environments'), data.get('random_seeds'))
    analyze_results(resultfolder)
