import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTIONS = [UP, RIGHT, DOWN, LEFT]


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
      policy: [S, A] shaped matrix representing the policy.
      env: OpenAI env. env.P represents the transition probabilities of the environment.
        env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
      theta: We stop evaluation once our value function change is less than theta for all states.
      discount_factor: gamma discount factor.

    Returns:
      Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s, action_space in env.P.items():
            v = V[s]

            sum_over_actions = 0
            for a in ACTIONS:
                pol = policy[s][a]

                sum_per_next_s_in_action = 0
                for transition_tuple in action_space[a]:
                    probability = transition_tuple[0]
                    next_state = transition_tuple[1]
                    reward = transition_tuple[2]
                    sum_per_next_s_in_action += probability * (reward + discount_factor * V[next_state])

                sum_over_actions += pol * sum_per_next_s_in_action
            V[s] = sum_over_actions
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
      env: The OpenAI envrionment.
      policy_eval_fn: Policy Evaluation function that takes 3 arguments:
        policy, env, discount_factor.
      discount_factor: Lambda discount factor.

    Returns:
      A tuple (policy, V).
      policy is the optimal policy, a matrix of shape [S, A] where each state s
      contains a valid probability distribution over actions.
      V is the value function for the optimal policy.

    """
    V = np.zeros(env.nS)
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # print(policy.shape)
    # print(V.shape)

    i = 0
    while True:
        V = policy_eval_fn(policy, env, discount_factor)
        policy

        for x in range(env.MAX_X):
            for y in range(env.MAX_Y):
                if V[y * 4 + x] == i:
                    V[y * 4 + x] = i
                    if x < env.MAX_X - 1 and policy[y * 4 + x + 1] is not 0.25:  # if next is right -> go left
                        policy[y * 4 + x + 1] = [0, 0, 0, 1]
                    if x > 0 and policy[y * 4 + x - 1] is not 0.25:  # if next is left -> go right
                        policy[y * 4 + x - 1] = [0, 1, 0, 0]
                    if y < env.MAX_Y - 1 and policy[(y + 1) * 4 + x] is not 0.25:  # if next is bottom -> go up
                        policy[(y + 1) * 4 + x] = [1, 0, 0, 0]
                    if y > 0 and policy[(y - 1) * 4 + x] is not 0.25:  # if next is on top -> go down
                        policy[(y - 1) * 4 + x] = [0, 0, 1, 0]
        i -= 1
        if i == -4:
            break
    # print(V)
    # print(eval.shape)
    # print(policy)
    return policy, V
