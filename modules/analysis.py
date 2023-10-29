# Author: Mario Niemann

# using the detailled log data to create some higher level analysis

import pandas as pd

def analyze_results(resultfolder):
    # load the log data
    df = pd.read_parquet(resultfolder + '/exp_results.parquet')
    
    # keep only required columns
    df = df[['algorithm','environment','iteration','random_seed','episode_reward_mean','episode_reward_max','evaluation_reward_mean','entropy','num_env_steps_trained','num_env_steps_sampled','time_this_iter_s','time_total_s']]
    
    # define grouping of data and columns to be calculated
    res_df = df.groupby(['algorithm','environment','iteration']).agg({
        'random_seed' : [('seed_count', 'count')],
        'episode_reward_mean': [('reward_Mean', 'mean'), ('reward_StdDev', 'std')],
        'episode_reward_max': [('reward_max_mean', 'mean'), ('reward_max_Max', 'max')],
        'evaluation_reward_mean': [('evaluation_reward_Mean', 'mean'), ('evaluation_reward_StdDev', 'std')],
        'time_this_iter_s': [('duration_Mean', 'mean')],
        'time_total_s': [('time_total_Mean', 'mean')],
        'entropy': [('entropy_Mean', 'mean')],
        'num_env_steps_trained': [('steps_trained_Mean', 'mean')],
        'num_env_steps_sampled': [('steps_sampled_Mean', 'mean')]
    }).reset_index()
    
    # calculate standard error for error bands on reward
    res_df.columns = [col[1] if col[1] != '' else col[0] for col in res_df.columns.values]
    res_df['reward_StdError'] = res_df['reward_StdDev'] / (res_df['seed_count'] ** 0.5)
    res_df['reward_upperErrorBound'] = res_df['reward_Mean'] + res_df['reward_StdError']
    res_df['reward_lowerErrorBound'] = res_df['reward_Mean'] - res_df['reward_StdError']
    
    #print(res_df)

    # save to csv file
    res_df.to_csv(resultfolder + '/exp_analysis.csv', index=False, sep=';', decimal=',')