import re
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def export(out_path, out_dict):
    df = pd.DataFrame.from_dict(out_dict, orient="columns")
    df.to_csv(out_path)
    print(f"Exported@{out_path}")


def average(out_arr):
    out_mean_dict = {}
    out_se_dict = {}
    for key in out_arr[0].keys():
        mean, se = key_average(key, out_arr)
        out_mean_dict[key] = mean
        out_se_dict[key] = se
    return out_mean_dict, out_se_dict


def key_average(key, out_arr):
    arr = []
    for out_dict in out_arr:
        arr.append(out_dict[key])
    arr = np.array(arr)
    return np.mean(arr, axis=0), stats.sem(arr, axis=0)


def create_upper_lower(mean_dict, se_dict):
    upper_dict = {}
    lower_dict = {}
    for key, value in mean_dict.items():
        upper_dict[key] = value + se_dict[key]
        lower_dict[key] = value - se_dict[key]
    return upper_dict, lower_dict


def export_curve(out_path, mean_dict, se_dict, upper_dict, lower_dict):
    columns = []
    values = []
    for key in mean_dict.keys():
        columns.append(key+"_mean")
        values.append(mean_dict[key])
        columns.append(key+"_se")
        values.append(se_dict[key])
        columns.append(key+"_lower")
        values.append(lower_dict[key])
        columns.append(key+"_upper")
        values.append(upper_dict[key])
    values = np.array(values).T
    df = pd.DataFrame(values, columns=columns)
    df.to_csv(out_path)
    print(f"Exported@{out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--label', type=str)
    parser.add_argument('--window_size', type=int, default=20)
    args = parser.parse_args()

    # define the names of the models you want to plot and the longest episodes you want to show
    # models = ['CADRL', 'CADRL-DTA', 'SARL', 'SARL-DTA']
    max_episodes = 10000
    val_arr, test_arr, train_arr = [], [], []
    for i, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()
        dir_name = os.path.dirname(log_file)
        # r"nav length: (?P<length>[-+]?\d+.\d+), " \

        # val_pattern = r"VAL   has \D+\s*\D*, avg human time: (?P<ht>\d+.\d+), nav time: (?P<nt>\d+.\d+), " \
        #               r"nav path length: (?P<npl>\d+.\d+), jerk: (?P<jerk>[-+]?\d+.\d+), total reward: (?P<tr>[-+]?\d+.\d+)"
        # val_human_time = []
        # val_nav_time = []
        # val_nav_path_length = []
        # val_jerk = []
        # val_total_reward = []
        # for r in re.findall(val_pattern, log):
        #     val_human_time.append(float(r[0]))
        #     val_nav_time.append(float(r[1]))
        #     val_nav_path_length.append(float(float(r[2])))
        #     val_jerk.append(float(r[3]))
        #     val_total_reward.append(float(r[4]))
        # val_out_dict = {
        #     "human_time": val_human_time,
        #     "nav_time": val_nav_time,
        #     "nav_path_length": val_nav_path_length,
        #     "jerk": val_jerk,
        #     "total_reward": val_total_reward
        # }
        # val_arr.append(val_out_dict)
        # val_path = os.path.join(dir_name, "val_results.csv")
        # export(val_path, val_out_dict)
                      
        test_pattern = r"TEST   in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                      r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                      r"total reward: (?P<reward>[-+]?\d+.\d+)"
        test_episode = []
        test_sr = []
        test_cr = []
        test_time = []
        test_reward = []
        for r in re.findall(test_pattern, log):
            test_episode.append(int(r[0]))
            test_sr.append(float(r[1]))
            test_cr.append(float(r[2]))
            test_time.append(float(r[3]))
            test_reward.append(float(r[4]))

        test_out_dict = {
            "episode": test_episode,
            "success_rate": test_sr,
            "collision_rate": test_cr,
            "nav_time": test_time,
            "total_reward": test_reward
        }
        test_arr.append(test_out_dict)
        test_path = os.path.join(dir_name, "test_results.csv")
        export(test_path, test_out_dict)

        # r"nav length: (?P<length>[-+]?\d+.\d+), " \
        
        train_pattern = r"TRAIN in episode (?P<episode>\d+) avg human time: (?P<ht>\d+.\d+) has success rate: (?P<sr>[0-1].\d+), " \
                        r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                        r"total reward: (?P<reward>[-+]?\d+.\d+)"
        train_episode = []
        train_sr = []
        train_cr = []
        train_time = []
        train_reward = []
        for r in re.findall(train_pattern, log):
            train_episode.append(int(r[0]))
            train_sr.append(float(r[2]))
            train_cr.append(float(r[3]))
            train_time.append(float(r[4]))
            train_reward.append(float(r[5]))
        train_episode = train_episode[:max_episodes]
        train_sr = train_sr[:max_episodes]
        train_cr = train_cr[:max_episodes]
        train_time = train_time[:max_episodes]
        train_reward = train_reward[:max_episodes]
        # "episode": train_episode,
        train_sr_smooth = running_mean(train_sr, args.window_size)
        train_cr_smooth = running_mean(train_cr, args.window_size)
        train_time_smooth = running_mean(train_time, args.window_size)
        train_reward_smooth = running_mean(train_reward, args.window_size)
        train_out_dict = {
            "success_rate": train_sr_smooth,
            # "success_rate_smooth": train_sr_smooth,
            "collision_rate": train_cr_smooth,
            # "collision_rate_smooth": train_cr_smooth,
            "nav_time": train_time_smooth,
            # "nav_time_smooth": train_time_smooth,
            "total_reward": train_reward_smooth,
            # "total_reward_smooth": train_reward_smooth
        }
        train_arr.append(train_out_dict)
        train_path = os.path.join(dir_name, "train_results.csv")
        export(train_path, train_out_dict)
    # average_val, se_val = average(val_arr)
    # upper_val, lower_val = create_upper_lower(average_val, se_val)
    average_test, se_test = average(test_arr)
    upper_test, lower_test = create_upper_lower(average_test, se_test)
    average_train, se_train = average(train_arr)
    upper_train, lower_train = create_upper_lower(average_train, se_train)
    # outpath = os.path.join("data", "analysis", f"{args.label}val_learning_curves.csv")
    # export_curve(outpath, average_val, se_val, upper_val, lower_val)
    # outpath = os.path.join("data", "analysis", f"{args.label}test_learning_curves.csv")
    # export_curve(outpath, average_test, se_test, upper_test, lower_test)

    outpath = os.path.join(
        "data",
        "analysis",
        f"{args.label}train_learning_curves.csv"
    )
    export_curve(outpath, average_train, se_train, upper_train, lower_train)


if __name__ == '__main__':
    main()
