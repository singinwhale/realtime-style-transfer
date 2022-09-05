import numpy as np


def get_stats(tensor: np.array):
    stats = {
        'mean': np.mean(tensor),
        'var': np.var(tensor),
        'min': np.min(tensor),
        'max': np.max(tensor),
    }

    return stats


def print_stat_comparison(stat_a_name, stat_a, stat_b_name, stat_b):
    print(f"{'Stat':10}\t\t{stat_a_name:20}\t\t{stat_b_name:20}\t\t{'Diff':20}")
    for stat in stat_a.keys():
        print(
            f"{stat:10}\t\t{stat_a[stat]: 20}"
            f"\t\t{stat_b[stat]: 20}"
            f"\t\t{stat_a[stat] - stat_b[stat]: 20}"
        )


def print_stats(stat_data):
    print(f"{'Name':10}\t\t{'Value':20}")
    for stat_name in stat_data.keys():
        print(
            f"{stat_name:10}\t\t{stat_data[stat_name]: 20}"
        )
