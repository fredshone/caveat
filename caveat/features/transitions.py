from pandas import DataFrame, Series

from caveat.features.utils import weighted_features


def extract_transitions(acts: Series) -> list[str]:
    """
    Extracts the transitions from the given Series of activities.

    Args:
        acts (Series): A Series containing the activities.

    Returns:
        list[str]: A list of transitions.
    """
    return [f"{a}>{b}" for a, b in zip(acts, acts[1:])]


def extract_transition_counts(acts: Series) -> Series:
    """
    Extracts counts of transitions from the given Series of activities.

    Args:
        acts (Series): A Series containing the activities.

    Returns:
        Series: counts of transitions
    """
    transitions = extract_transitions(acts)
    return Series(transitions).value_counts()


def transitions_by_act(population: DataFrame) -> dict[str, list]:
    transitions = population.reset_index()
    transitions = transitions.set_index(["index", "pid"])
    transitions.act = transitions.act.astype(str)
    transitions = transitions.act + ">" + transitions.act.shift(-1)
    transitions = transitions.drop(transitions.groupby("pid").tail(1).index)
    transitions = (
        transitions.groupby("pid")
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
        .to_dict(orient="list")
    )
    return weighted_features(transitions)


def transition_3s_by_act(population: DataFrame) -> dict[str, list]:
    transitions = population.reset_index()
    transitions = transitions.set_index(["index", "pid"])
    transitions.act = transitions.act.astype(str)
    transitions = (
        transitions.act
        + ">"
        + transitions.act.shift(-1)
        + ">"
        + transitions.act.shift(-2)
    )
    transitions = transitions.drop(transitions.groupby("pid").tail(2).index)
    transitions = (
        transitions.groupby("pid")
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
        .to_dict(orient="list")
    )
    return weighted_features(transitions)


def tour(acts: Series) -> str:
    """
    Extracts the tour from the given Series of activities.

    Args:
        acts (Series): A Series containing the activities.

    Returns:
        str: A string representation of the tour.
    """
    return ">".join(acts.str[0])


def full_sequences(population: DataFrame) -> dict[str, list]:
    transitions = population.reset_index()
    transitions = transitions.set_index(["index", "pid"])
    transitions.act = transitions.act.astype(str)
    transitions = transitions.groupby("pid").act.apply(tour)
    transitions = (
        transitions.groupby("pid")
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
        .to_dict(orient="list")
    )
    return weighted_features(transitions)


# def collect_sequence(acts: Series) -> str:
#     return ">".join(acts)


# def sequence_probs(population: DataFrame) -> DataFrame:
#     """
#     Calculates the sequence probabilities in the given population DataFrame.

#     Args:
#         population (DataFrame): A DataFrame containing the population data.

#     Returns:
#         DataFrame: A DataFrame containing the probability of each sequence.
#     """
#     metrics = (
#         population.groupby("pid")
#         .act.apply(collect_sequence)
#         .value_counts(normalize=True)
#     )
#     metrics = metrics.sort_values(ascending=False)
#     metrics.index = MultiIndex.from_tuples(
#         [("sequence rate", acts) for acts in metrics.index]
#     )
#     return metrics


# def transition_rates(population: DataFrame) -> Series:
#     """
#     Calculates the transition rates per person in the given population DataFrame.

#     Args:
#         population (DataFrame): A DataFrame containing the population data.

#     Returns:
#         Series: A Series containing the rate of occurance of each transition per person.
#     """
#     transitions = {}
#     for acts in population.groupby("pid").act.apply(list):
#         for i in range(len(acts) - 1):
#             t = "->".join(acts[i : i + 2])
#             transitions[t] = transitions.get(t, 0) + 1
#     transitions = Series(transitions).sort_values(ascending=False)
#     transitions /= population.pid.nunique()
#     transitions = transitions.sort_values(ascending=False)
#     transitions.index = MultiIndex.from_tuples(
#         [("transition rate", acts) for acts in transitions.index]
#     )
#     return transitions
