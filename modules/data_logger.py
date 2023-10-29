# Author: Mario Niemann

# Creating a custom callback for logging the created data into a parquet-file

from ray.tune import Callback
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

class CustomCallbacks(Callback):
    def __init__(self, filepath, logdata, algorithm, random_seed):
        self.writer = None
        self.filepath = filepath
        self.algorithm = algorithm
        self.environment = logdata['environment']
        self.random_seed = random_seed

    def __del__(self):
        if self.writer:
            self.writer.close()

    # log data after every trial (certain amount of steps sampled for ray.tune)
    def on_trial_result(self, iteration, trials, trial, result, **kwargs):
        #print(result)
        data={}
        data['algorithm'] = self.algorithm
        data['environment'] = self.environment
        data['random_seed'] = self.random_seed
        data['iteration'] = result.get('training_iteration')
        data['actor_loss'] = float('nan')
        data['critic_loss'] = float('nan')
        data['mean_q'] = float('nan')
        data['max_q'] = float('nan')
        data['min_q'] = float('nan')
        data['total_loss'] = float('nan')
        data['policy_loss'] = float('nan')
        data['vf_loss'] = float('nan')
        data['entropy'] = float('nan')
        if result['info']['learner'].get('default_policy'):
            if result['info']['learner']['default_policy'].get('learner_stats'):
                data['actor_loss'] = result['info']['learner']['default_policy']['learner_stats'].get('actor_loss') if result['info']['learner']['default_policy']['learner_stats'].get('actor_loss') else float('nan')
                data['critic_loss'] = result['info']['learner']['default_policy']['learner_stats'].get('critic_loss') if result['info']['learner']['default_policy']['learner_stats'].get('critic_loss') else float('nan')
                data['mean_q'] = result['info']['learner']['default_policy']['learner_stats'].get('mean_q') if result['info']['learner']['default_policy']['learner_stats'].get('mean_q') else float('nan')
                data['max_q'] = result['info']['learner']['default_policy']['learner_stats'].get('max_q') if result['info']['learner']['default_policy']['learner_stats'].get('max_q') else float('nan')
                data['min_q'] = result['info']['learner']['default_policy']['learner_stats'].get('min_q') if result['info']['learner']['default_policy']['learner_stats'].get('min_q') else float('nan')
            else:
                data['total_loss'] = result['info']['learner']['default_policy'].get('total_loss') if result['info']['learner']['default_policy'].get('total_loss') else float('nan')
                data['policy_loss'] = result['info']['learner']['default_policy'].get('policy_loss') if result['info']['learner']['default_policy'].get('policy_loss') else float('nan')
                data['vf_loss'] = result['info']['learner']['default_policy'].get('vf_loss') if result['info']['learner']['default_policy'].get('vf_loss') else float('nan')
                data['entropy'] = result['info']['learner']['default_policy'].get('entropy') if result['info']['learner']['default_policy'].get('entropy') else float('nan')
        data['episode_reward_mean'] = result.get('episode_reward_mean')
        data['episode_reward_min'] = result.get('episode_reward_min')
        data['episode_reward_max'] = result.get('episode_reward_max')
        data['episode_len_mean'] = result.get('episode_len_mean')
        data['episodes_this_iter'] = result.get('episodes_this_iter')
        data['mean_raw_obs_processing_ms'] = result['sampler_perf'].get('mean_raw_obs_processing_ms') if result['sampler_perf'].get('mean_raw_obs_processing_ms') else 0.0
        data['mean_inference_ms'] = result['sampler_perf'].get('mean_inference_ms') if result['sampler_perf'].get('mean_inference_ms') else 0.0
        data['mean_action_processing_ms'] = result['sampler_perf'].get('mean_action_processing_ms') if result['sampler_perf'].get('mean_action_processing_ms') else 0.0
        data['mean_env_wait_ms'] = result['sampler_perf'].get('mean_env_wait_ms') if result['sampler_perf'].get('mean_env_wait_ms') else 0.0
        data['mean_env_render_ms'] = result['sampler_perf'].get('mean_env_render_ms') if result['sampler_perf'].get('mean_env_render_ms') else 0.0
        data['training_iteration_time_ms'] = result['timers'].get('training_iteration_time_ms') if result['timers'].get('training_iteration_time_ms') else 0.0
        data['sample_time_ms'] = result['timers'].get('sample_time_ms') if result['timers'].get('sample_time_ms') else 0.0
        data['load_time_ms'] = result['timers'].get('load_time_ms') if result['timers'].get('load_time_ms') else 0.0
        data['load_throughput'] = result['timers'].get('load_throughput') if result['timers'].get('load_throughput') else 0.0
        data['learn_time_ms'] = result['timers'].get('learn_time_ms') if result['timers'].get('learn_time_ms') else 0.0
        data['learn_throughput'] = result['timers'].get('learn_throughput') if result['timers'].get('learn_throughput') else 0.0
        data['target_net_update_time_ms'] = result['timers'].get('target_net_update_time_ms') if result['timers'].get('target_net_update_time_ms') else 0.0
        data['synch_weights_time_ms'] = result['timers'].get('synch_weights_time_ms') if result['timers'].get('synch_weights_time_ms') else 0.0
        data['num_env_steps_sampled'] = result['counters'].get('num_env_steps_sampled') if result['counters'].get('num_env_steps_sampled') else 0
        data['num_env_steps_trained'] = result['counters'].get('num_env_steps_trained') if result['counters'].get('num_env_steps_trained') else 0
        data['num_agent_steps_sampled'] = result['counters'].get('num_agent_steps_sampled') if result['counters'].get('num_agent_steps_sampled') else 0
        data['num_agent_steps_trained'] = result['counters'].get('num_agent_steps_trained') if result['counters'].get('num_agent_steps_trained') else 0
        data['last_target_update_ts'] = result['counters'].get('last_target_update_ts') if result['counters'].get('last_target_update_ts') else 0
        data['num_target_updates'] = result['counters'].get('num_target_updates') if result['counters'].get('num_target_updates') else 0
        data['evaluation_reward_mean'] = result['evaluation']['episode_reward_mean'] if result.get('evaluation') else float('nan')
        data['time_this_iter_s'] = result['time_this_iter_s']
        data['time_total_s'] = result['time_total_s']
        
        for key in data:
            data[key] = [data[key]]
        #print(data)
        self.append_to_parquet_table(data)

    # append new dataset to parquet data file
    # TODO: in case there is an error during the writing process, the file might be corrupted
    # Therefore using a temp-file in between and copy this could be more secure
    def append_to_parquet_table(self, dataframe):
        table = pa.Table.from_pydict(dataframe)
        if self.writer is None:
            content = None
            if os.path.isfile(self.filepath):
                content = pq.read_table(self.filepath)
            self.writer = pq.ParquetWriter(self.filepath, table.schema)
            if content:
                self.writer.write_table(table=content)
        self.writer.write_table(table=table)
