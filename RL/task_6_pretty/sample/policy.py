import numpy as np


def epsilon_greedy(actions: np.ndarray, epsilon: float):
    """
        Args:
          actions: A list of actions returned for a specific state on the Q-values
          epsilon: The probability to select a random action . float between 0 and 1.

        Returns:
          the index of the action to take
    """

    probabilities = np.ones(len(actions), dtype=float) * epsilon / (len(actions) - 1)

    max_action_value = np.max(actions)
    best_action = np.random.choice(np.flatnonzero(actions == max_action_value))  # don't use argmax ->
    # if multiple actions have max-value, choose randomly instead of always using the first like argmax

    probabilities[best_action] = 1.0 - epsilon
    return np.random.choice(np.arange(len(probabilities)), p=probabilities)
