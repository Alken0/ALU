import numpy as np


def update(Q, state: int, action: int, reward: float, next_state: int, done: bool, discount_factor: float, alpha: float) -> None:
    best_next_action_value = np.max(Q[next_state])
    td_target = reward + (1 - done) * discount_factor * best_next_action_value
    td_delta = td_target - Q[state][action]
    Q[state][action] += alpha * td_delta
