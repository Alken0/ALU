import sys
from collections import defaultdict, namedtuple
from typing import Dict

from sample import GridworldEnv, policy, td
import itertools
import numpy as np

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def dyna_q_learning(env: GridworldEnv, num_episodes: int, discount_factor=0.95, alpha=0.1, epsilon=0.1, n=50):
    """
    Dyna-Q-Learning algorithm: Off-policy Model-based TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
      env: environment.
      num_episodes: Number of episodes to run for.
      discount_factor: Lambda time discount factor.
      alpha: TD learning rate.
      epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
      A tuple (Q, episode_lengths).
      Q is the optimal action-value function, a dictionary mapping state -> action values.
      stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q: Dict[np.ndarray[float]] = defaultdict(lambda: np.zeros(env.nA))
    # The model.
    # A nested dictionary that maps state -> (action -> (next state, reward, terminal flag)).
    M: Dict[np.ndarray[(int, float, bool)]] = defaultdict(lambda: np.zeros((env.nA, 3)))
    observed_sa = []

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 1 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            # Take a step
            action = policy.epsilon_greedy(Q[state], epsilon)
            next_state, reward, done = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            td.update(Q, state, action, reward, next_state, done, discount_factor, alpha)
            M[state][action] = [next_state, reward, done]

            if (state, action) not in observed_sa:
                observed_sa.append((state, action))

            for _ in range(n):
                img_state, img_action = observed_sa[np.random.choice(len(observed_sa))]
                img_next_state, img_reward, img_done = M[img_state][img_action]
                td.update(Q, img_state, img_action, img_reward, img_next_state, img_done, discount_factor, alpha)

            if done:
                break

            state = next_state

    return Q, stats
