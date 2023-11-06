from pandas import DataFrame, MultiIndex, Series


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
