import sys
from collections import defaultdict, namedtuple
from gridworld import GridworldEnv
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
      Q: A dictionary that maps from state -> action-values.
        Each value is a numpy array of length nA (see below)
      epsilon: The probability to select a random action . float between 0 and 1.
      nA: Number of actions in the environment.

    Returns:
      A function that takes the observation as an argument and returns
      the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.random.choice(np.flatnonzero(Q[observation] == Q[observation].max()))
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def action(policy, s):
    action_probabilities = policy(s)
    return np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)


def dyna_q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, n=5):
    """
    Dyna-Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
      env: environment.
      num_episodes: Number of episodes to run for.
      discount_factor: Lambda time discount factor.
      alpha: TD learning rate.
      epsilon: Chance the sample a random action. Float betwen 0 and 1.
      n: number of planning steps

    Returns:
      A tuple (Q, episode_lengths).
      Q is the optimal action-value function, a dictionary mapping state -> action values.
      stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.nA))

    # The model.
    # A nested dictionary that maps state -> (action -> (next state, reward, terminal flag)).
    M = defaultdict(lambda: np.zeros((env.nA, 3)))

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 1 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        s = env.reset()
        previous = []
        for t in itertools.count():
            a = action(policy, s)

            if (s, a) not in previous:
                previous.append((s, a))  # add to list used in 2nd part

            # do one step
            r, ns, done = env.step(a)

            # update statistics
            stats.episode_rewards[i_episode] += r
            stats.episode_lengths[i_episode] = t

            # update Q and M
            Q[s][a] = Q[s][a] + alpha * (
                    r + (1 - done) * discount_factor * np.max(Q[ns]) - Q[s][a])
            M[s][a] = (ns, r, False)

            for i in range(n):
                rs, ra = previous[np.random.choice(len(previous))]
                rr, rns, _ = M[rs][ra]
                Q[rs][ra] = Q[rs][ra] + alpha * (rr + discount_factor * np.max(Q[rns]) - Q[rs][ra])

            if done:
                break

            s = ns  # set state for next episode

    return Q, stats


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
    np.random.seed(0)
    env = GridworldEnv()
    Q, stats = dyna_q_learning(env, 100, n=50)

    print("")
    for k, v in Q.items():
        print("%s: %s" % (k, v.tolist()))
    plot_episode_stats(stats)
