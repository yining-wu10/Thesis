import numpy as np
from copy import deepcopy

# Optimality Gap Evaluation
#

"""
    Select an action using the learned MRL policy.
    
    Parameters
    ----------
    model:
        Trained MRL model. Must contain:
        - model.m.predict()
        - model.pi
    state:
        Current simulator state.
    state_to_features:
        Function that converts simulator state into the feature vector
        expected by model.m.predict().
        If None, state itself is used as the feature vector.
    
    Returns
    -------
    action: Action selected by the learned MRL policy.
    """
def mrl_policy_action(model, state, state_to_features=None):

    if state_to_features is None:
        features = state
    else:
        features = state_to_features(state)

    cluster = int(model.m.predict([features])[0])
    action = model.pi[cluster]

    return action


"""
    Roll out one policy in the sepsis simulator.

    Parameters
    ----------
    initial_state:
        Initial patient state.
    policy_fn:
        Function mapping state -> action.
    env_step:
        Simulator transition function:
            next_state, reward, done, info = env_step(state, action)
    t_max:
        Maximum number of simulation steps.
    gamma:
        Discount factor.
    Returns
    -------
    total_return:
        Discounted cumulative return.
    """

def rollout_policy(
    initial_state,
    policy_fn,
    env_step,
    t_max,
    gamma=1.0,
):
    state = deepcopy(initial_state)
    total_return = 0.0
    done = False

    for t in range(t_max):
        if done:
            break

        action = policy_fn(state)
        next_state, reward, done, info = env_step(state, action)

        total_return += (gamma ** t) * reward
        state = next_state

    return total_return


def evaluate_model_gap(
    model,
    initial_states,
    env_step,
    baseline_policy,
    t_max,
    gamma=1.0,
    state_to_features=None,
    use_median=False,
):
    """
    Evaluate one MRL model against a baseline policy.

    Parameters
    ----------
    model:
        One trained MRL model.

    initial_states:
        List or array of initial patient states, usually all patients' time=0 states.

    env_step:
        Sepsis simulator step function.

    baseline_policy:
        Reference policy, e.g. true optimal policy, physician policy, random policy.

    t_max:
        Maximum episode length.

    gamma:
        Discount factor.

    state_to_features:
        Optional state conversion function.

    use_median:
        If True, use median return instead of mean return.

    Returns
    -------
    gap:
        Absolute value difference between MRL policy and baseline policy.
    """

    if model.pi is None:
        model.solve_MDP(gamma=gamma, epsilon=1e-4)

    def learned_policy(state):
        return mrl_policy_action(
            model=model,
            state=state,
            state_to_features=state_to_features,
        )

    learned_returns = []
    baseline_returns = []

    for s0 in initial_states:
        G_learned = rollout_policy(
            initial_state=s0,
            policy_fn=learned_policy,
            env_step=env_step,
            t_max=t_max,
            gamma=gamma,
        )

        G_baseline = rollout_policy(
            initial_state=s0,
            policy_fn=baseline_policy,
            env_step=env_step,
            t_max=t_max,
            gamma=gamma,
        )

        learned_returns.append(G_learned)
        baseline_returns.append(G_baseline)

    if use_median:
        gap = abs(np.median(learned_returns) - np.median(baseline_returns))
    else:
        gap = abs(np.mean(learned_returns) - np.mean(baseline_returns))

    return gap


def value_diff_sepsis(
    models,
    initial_states,
    env_step,
    baseline_policy,
    t_max,
    gamma=1.0,
    state_to_features=None,
    use_median=False,
):
    """
    Compute value gaps for multiple MRL models.

    Parameters
    ----------
    models:
        List of trained MRL models.

    initial_states:
        Initial patient states used for evaluation.

    env_step:
        Sepsis simulator step function:
            next_state, reward, done, info = env_step(state, action)

    baseline_policy:
        Reference policy:
            action = baseline_policy(state)

    t_max:
        Maximum episode length.

    gamma:
        Discount factor.

    state_to_features:
        Optional converter from simulator state to MRL feature vector.

    use_median:
        Whether to use median returns instead of mean returns.

    Returns
    -------
    v_alg:
        List of value gaps, one per model.
    """

    v_alg = []

    for model in models:
        gap = evaluate_model_gap(
            model=model,
            initial_states=initial_states,
            env_step=env_step,
            baseline_policy=baseline_policy,
            t_max=t_max,
            gamma=gamma,
            state_to_features=state_to_features,
            use_median=use_median,
        )

        v_alg.append(gap)

    return v_alg
