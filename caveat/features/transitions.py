from pandas import DataFrame, MultiIndex, Series


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


def raw_transition_rates(population: DataFrame) -> dict[str, list]:
    rates = (
        population.groupby("pid")
        .act.apply(extract_transition_counts)
        .unstack()
        .fillna(0)
    )
    return rates.to_dict(orient="list")


def transition_rates(population: DataFrame) -> Series:
    """
    Calculates the transition rates per person in the given population DataFrame.

    Args:
        population (DataFrame): A DataFrame containing the population data.

    Returns:
        Series: A Series containing the rate of occurance of each transition per person.
    """
    transitions = {}
    for acts in population.groupby("pid").act.apply(list):
        for i in range(len(acts) - 1):
            t = "->".join(acts[i : i + 2])
            transitions[t] = transitions.get(t, 0) + 1
    transitions = Series(transitions).sort_values(ascending=False)
    transitions /= population.pid.nunique()
    transitions = transitions.sort_values(ascending=False)
    transitions.index = MultiIndex.from_tuples(
        [("transition rate", acts) for acts in transitions.index]
    )
    return transitions
