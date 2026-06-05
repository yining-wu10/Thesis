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


"""
?
Compute discounted return for each generated trajectory.
"""
def compute_returns(iter_rewards, iter_lengths, gamma=1.0):
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



"""
Use DataGenerator to generate trajectories under policy_array and estimate mean policy value.
Returns:
   value: mean trajectory return
   returns: return of each simulated trajectory
"""
def estimate_policy_value(
    policy_array,
    num_iters,
    max_num_steps,
    gamma=1.0,
    policy_idx_type="obs",
    output_state_idx_type="obs",
    p_diabetes=0.2,
    use_tqdm=False,
    tqdm_desc="",
):
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
        use_tqdm=use_tqdm,
        tqdm_desc=tqdm_desc,
    )

    returns = compute_returns(
        iter_rewards=iter_rewards,
        iter_lengths=iter_lengths,
        gamma=gamma,
    )

    return np.mean(returns), returns



"""
Compute value gaps between MRL policies and a baseline policy.

Parameters
----------
    models: List of trained MRL models.
    baseline_pi: Deterministic baseline policy - baseline_pi[state_idx] = action_idx
    num_states:
        Number of states corresponding to idx_type:
            idx_type="obs"      -> State.NUM_OBS_STATES
            idx_type="proj_obs" -> State.NUM_PROJ_OBS_STATES
            idx_type="full"     -> State.NUM_FULL_STATES
    num_iters: Number of simulated trajectories per policy.
    max_num_steps: Maximum trajectory length.
    idx_type:
        Determines both:
            1. how state_idx is decoded into features
            2. which state index type DataGenerator uses for policy lookup
    diabetic_idx:
        Used only for decoding obs/proj_obs state indices into State objects.
        Ignored for full state indices.
    gamma: Discount factor.
    p_diabetes: Initial diabetes probability used by DataGenerator.
    use_median: If True, compare median returns instead of mean returns.

Returns
-------
    v_alg: List of value gaps, one per MRL model.
    """
def value_diff(
    models,
    baseline_pi,
    num_states,
    num_iters,
    max_num_steps,
    idx_type="obs",
    diabetic_idx=0,
    gamma=1.0,
    p_diabetes=0.2,
    use_median=False,
    use_tqdm=False,
):
    assert idx_type in ["obs", "proj_obs", "full"]

    policy_idx_type = idx_type
    output_state_idx_type = idx_type
    num_actions = Action.NUM_ACTIONS_TOTAL

    baseline_policy_array = deterministic_pi_to_prob_array(
        pi=baseline_pi,
        num_actions=num_actions,
    )

    baseline_value, baseline_returns = estimate_policy_value(
        policy_array=baseline_policy_array,
        num_iters=num_iters,
        max_num_steps=max_num_steps,
        gamma=gamma,
        policy_idx_type=policy_idx_type,
        output_state_idx_type=output_state_idx_type,
        p_diabetes=p_diabetes,
        use_tqdm=use_tqdm,
        tqdm_desc="baseline policy",
    )

            

    v_alg = []

    for i, model in enumerate(models):

        if model.pi is None:
            model.solve_MDP(gamma=gamma, epsilon=1e-4)

        mrl_policy_array = mrl_model_to_prob_array(
            model=model,
            num_states=num_states,
            idx_type=idx_type,
            diabetic_idx=diabetic_idx,
            num_actions=num_actions,
        )

        mrl_value, mrl_returns = estimate_policy_value(
            policy_array=mrl_policy_array,
            num_iters=num_iters,
            max_num_steps=max_num_steps,
            gamma=gamma,
            policy_idx_type=policy_idx_type,
            output_state_idx_type=output_state_idx_type,
            p_diabetes=p_diabetes,
            use_tqdm=use_tqdm,
            tqdm_desc=f"MRL model {i}",
        )

        if use_median:
            gap = abs(np.median(mrl_returns) - np.median(baseline_returns))
        else:
            gap = abs(mrl_value - baseline_value)

        v_alg.append(gap)

    return v_alg
