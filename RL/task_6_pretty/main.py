import numpy as np
import sample
from matplotlib import pyplot as plt
import pandas as pd


def main():
    np.random.seed(0)
    env = sample.GridworldEnv()
    Q, stats = sample.dyna_q_learning(env, 100, n=50)

    print("")
    for k, v in Q.items():
        print("%s: %s" % (k, v.tolist()))
    plot_episode_stats(stats)


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close()
    else:
        plt.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close()
    else:
        plt.show()

    return fig1, fig2


if __name__ == "__main__":
    main()
