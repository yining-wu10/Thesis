import numpy as np
from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.DataGenerator import DataGenerator



"""
Decode simulator state index into MRL classifier features.
idx_type="obs":
            return 7-dimensional observed features:
            [hr, sysbp, percoxyg, glucose, antibiotic, vaso, vent]
idx_type="proj_obs":
            return 7-dimensional observed features.
            In State.py, glucose is set to normal level 2.
idx_type="full":
            return 8-dimensional full features:
            [diabetic_idx, hr, sysbp, percoxyg, glucose, antibiotic, vaso, vent]

Parameters
----------
state_idx: Integer state index.
idx_type: "obs", "proj_obs", or "full".
diabetic_idx:
        Only used when idx_type is "obs" or "proj_obs".
        Ignored when idx_type is "full".
"""
def state_idx_to_features(state_idx, idx_type="obs", diabetic_idx=0):
    assert idx_type in ["obs", "proj_obs", "full"]

    if idx_type == "full":
        state = State(
            state_idx=int(state_idx),
            idx_type="full",
            diabetic_idx=None,
        )

        obs_features = state.get_state_vector().astype(int)

        return np.concatenate([
            np.array([int(state.diabetic_idx)]),
            obs_features,
        ])

    state = State(
        state_idx=int(state_idx),
        idx_type=idx_type,
        diabetic_idx=int(diabetic_idx),
    )

    return state.get_state_vector().astype(int)


"""
Convert deterministic policy:
        pi[state] = action_idx
into probability policy:
        policy_array[state, action] = probability
DataGenerator requires a probability vector over actions for each state.
"""
def deterministic_pi_to_prob_array(pi, num_actions=8):
    num_states = len(pi)
    policy_array = np.zeros((num_states, num_actions))

    for s in range(num_states):
        action = int(pi[s])
        policy_array[s, action] = 1.0

    return policy_array




"""
Convert learned MRL model policy into DataGenerator-compatible policy array.
For each original simulator state:
        state_idx
        -> features via state_idx_to_features()
        -> cluster via model.m.predict()
        -> action via model.pi[cluster]
        -> one-hot probability distribution over actions
"""
def mrl_model_to_prob_array(
    model,
    num_states,
    idx_type="obs",
    diabetic_idx=0,
    num_actions=8,
):
    policy_array = np.zeros((num_states, num_actions))

    for state_idx in range(num_states):
        features = state_idx_to_features(
            state_idx=state_idx,
            idx_type=idx_type,
            diabetic_idx=diabetic_idx,
        )

        cluster = int(model.m.predict([features])[0])
        action_idx = int(model.pi[cluster])
        policy_array[state_idx, action_idx] = 1.0

    return policy_array


def compute_returns(iter_rewards, iter_lengths, gamma=1.0):
    """
    Compute discounted return for each generated trajectory.
    """

    returns = []

    num_iters = iter_rewards.shape[0]

    for i in range(num_iters):
        T_i = int(iter_lengths[i, 0])
        rewards_i = iter_rewards[i, :T_i, 0]

        G = 0.0
        for t, reward in enumerate(rewards_i):
            G += (gamma ** t) * reward

        returns.append(G)

    return np.array(returns)


def estimate_policy_value(
    policy_array,
    num_iters,
    max_num_steps,
    gamma=1.0,
    policy_idx_type="obs",
    output_state_idx_type="obs",
    p_diabetes=0.2,
):
    """
    Use DataGenerator to generate trajectories under policy_array
    and estimate mean policy value.
    """

    dgen = DataGenerator()

    (
        iter_states,
        iter_actions,
        iter_lengths,
        iter_rewards,
        iter_component,
        emp_tx_mat,
        emp_r_mat,
    ) = dgen.simulate(
        num_iters=num_iters,
        max_num_steps=max_num_steps,
        policy=policy_array,
        policy_idx_type=policy_idx_type,
        p_diabetes=p_diabetes,
        output_state_idx_type=output_state_idx_type,
        use_tqdm=False,
    )

    returns = compute_returns(
        iter_rewards=iter_rewards,
        iter_lengths=iter_lengths,
        gamma=gamma,
    )

    return np.mean(returns), returns


def value_diff_sepsis_datagen(
    models,
    baseline_pi,
    num_states,
    num_iters,
    max_num_steps,
    state_to_features=None,
    gamma=1.0,
    policy_idx_type="obs",
    output_state_idx_type="obs",
    p_diabetes=0.2,
    use_median=False,
):
    """
    Compute policy value gaps using DataGenerator.

    baseline_pi:
        deterministic policy vector:
            baseline_pi[state] = action_idx

    Returns:
        v_alg:
            list of value gaps, one per MRL model
    """

    v_alg = []

    baseline_policy_array = deterministic_pi_to_prob_array(
        pi=baseline_pi,
        num_actions=8,
    )

    baseline_value, baseline_returns = estimate_policy_value(
        policy_array=baseline_policy_array,
        num_iters=num_iters,
        max_num_steps=max_num_steps,
        gamma=gamma,
        policy_idx_type=policy_idx_type,
        output_state_idx_type=output_state_idx_type,
        p_diabetes=p_diabetes,
    )

    for model in models:

        if model.pi is None:
            model.solve_MDP(gamma=gamma, epsilon=1e-4)

        mrl_policy_array = mrl_model_to_prob_array(
            model=model,
            num_states=num_states,
            state_to_features=state_to_features,
            num_actions=8,
        )

        mrl_value, mrl_returns = estimate_policy_value(
            policy_array=mrl_policy_array,
            num_iters=num_iters,
            max_num_steps=max_num_steps,
            gamma=gamma,
            policy_idx_type=policy_idx_type,
            output_state_idx_type=output_state_idx_type,
            p_diabetes=p_diabetes,
        )

        if use_median:
            gap = abs(np.median(mrl_returns) - np.median(baseline_returns))
        else:
            gap = abs(mrl_value - baseline_value)

        v_alg.append(gap)

    return v_alg
