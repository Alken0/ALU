import numpy as np


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
        for s in range(env.nS):
            v = 0
            for a, a_probability in enumerate(policy[s]):
                for probability, next_state, reward, _ in env.P[s][a]:
                    v += a_probability * probability * (reward + discount_factor * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
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
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    i = 0
    while True:
        V = policy_eval_fn(policy, env, discount_factor)
        policy_stable = True
        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])

            action_values = np.zeroes(env.nA)
            for a in range(env.nA):
                for probability, next_state, reward, _ in env.P[s][a]:
                    action_values[a] += probability * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

            # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V
