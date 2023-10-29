# Author: Mario Niemann

# Using ray.tune to run the experiment
# As tune in general allows for testing through different hyperparameter, it could also be used
# to integrate more parts of the overall experiment into this part, but would make the setup 
# of this pipeline more complex

import ray
from ray import tune
from ray.tune import Callback

from modules.data_logger import CustomCallbacks
import modules.environments as exp_env

# init ray, set num_cpus and num_gpus to local environment
# TODO: make num_cpus and num_gpus configurable via config file
ray.init(num_cpus=8,num_gpus=1)
exp_env.registerEnvironments()


# experiment execution
def run_experiment(logdata, algorithm, environment, random_seed):

    # read out stop criteria for selected algorithm, if given in config file
    stop_criteria = environment.get('stop_criteria_timesteps')
    if stop_criteria:
        stop_value = stop_criteria.get(algorithm)
    stop_criteria = {
        'timesteps_total': stop_value if stop_value else 100000 # stop after given stop criteria or 100.000 samplesteps
    }

    analysis = tune.run(
        algorithm,                      # set selected algorithm
        config={
            "env": environment['env'],  # set environment
            "seed": random_seed,        # set random seed
            "num_gpus": 0.2,            # num_gpus per worker
            "num_workers": 5,           # set amount of parallel workers
            "framework": 'torch',       # set ML framework
            "always_attach_evaluation_results": True,                           # to make readout a bit simpler
            "evaluation_num_episodes": 10,                                      # run 10 episodes for evaluation
            "evaluation_interval": 5 if algorithm in ['TD3','DDPG'] else 10,    # evaluate every 10 iterations or every 5 for TD3 and DDPG
            "evaluation_parallel_to_training": True,                            # have seperate worker for evaluation
            "evaluation_num_workers": 1                                         # amount evaluation worker
        },
        stop=stop_criteria,             # stop criteria as defined above

        # set data logger as callback
        callbacks = [CustomCallbacks(filepath=f"{logdata['resultfolder']}/exp_results.parquet", logdata=logdata, algorithm=algorithm, random_seed=random_seed)],

        metric = environment['metric'], # Metric to optimize for (given in config file)
        mode = environment['mode']      # Maximize/minimize the metric (given in config file)
    )