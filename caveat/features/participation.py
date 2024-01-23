from numpy import array, ndarray
from pandas import DataFrame

from caveat.features.utils import weighted_features


def participation_prob_by_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    """
    Calculate the participations by activity for a given population.

    Args:
        population (DataFrame): The population data.

    Returns:
        dict[str, tuple[array, array]]: A dictionary containing the participation for each activity.
    """
    metrics = population.groupby(["pid", "act"], observed=False).size() > 0
    metrics = metrics.groupby("act", observed=False).sum().to_dict()
    n = population.pid.nunique()
    compressed = {}
    for k, v in metrics.items():
        compressed[k] = (array([0, 1]), array([(n - v), v]))
    return compressed


def participation_rates(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    rates = population.groupby("pid").act.count()
    return weighted_features({"all": rates.to_list()})


def participation_rates_by_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    rates = population.groupby("pid").act.value_counts().unstack().fillna(0)
    return weighted_features(rates.to_dict(orient="list"))


def participation_rates_by_seq_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    rates = actseq.groupby(population.pid).value_counts().unstack().fillna(0)
    return weighted_features(rates.to_dict(orient="list"))


def participation_rates_by_act_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    act_enum = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)
    rates = act_enum.groupby(population.pid).value_counts().unstack().fillna(0)
    return weighted_features(rates.to_dict(orient="list"))


def combinations_with_replacement(
    targets: list, length: int, prev_array=[]
) -> list[list]:
    """
    Returns all possible combinations of elements in the input array with replacement,
    where each combination has a length of tuple_length.

    Args:
        targets (list): The input array to generate combinations from.
        length (int): The length of each combination.
        prev_array (list, optional): The previous array generated in the recursion. Defaults to [].

    Returns:
        list: A list of all possible combinations of elements in the input array with replacement.
    """
    if len(prev_array) == length:
        return [prev_array]
    combs = []
    for i, val in enumerate(targets):
        prev_array_extended = prev_array.copy()
        prev_array_extended.append(val)
        combs += combinations_with_replacement(
            targets[i:], length, prev_array_extended
        )
    return combs


def calc_pair_prob(act_counts, pair):
    a, b = pair
    if a == b:
        return (act_counts[a] > 1).sum()
    return ((act_counts[a] > 0) & (act_counts[b] > 0)).sum()


def joint_participation_prob(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    """
    Calculate the participation prob for all pairs of activities in the given population.

    Args:
        population (pandas.DataFrame): A DataFrame containing the population data.

    Returns:
        pandas.Series: A Series containing the participation rate for all pairs of activities.
    """
    act_counts = (
        population.groupby("pid").act.value_counts().unstack(fill_value=0)
    )
    acts = list(population.act.unique())
    pairs = combinations_with_replacement(acts, 2)
    n = population.pid.nunique()
    metric = {}
    for pair in pairs:
        p = calc_pair_prob(act_counts, pair)
        metric["+".join(pair)] = (array([0, 1]), array([n - p, p]))

    return metric


def calc_pair_rate(act_counts, pair):
    a, b = pair
    if a == b:
        return ((act_counts[a] / 2).astype(int)).value_counts().to_dict()
    return (
        ((act_counts[[a, b]].min(axis=1) / 2).astype(int))
        .value_counts()
        .to_dict()
    )


def joint_participation_rate(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    """
    Calculate the participation rate for all pairs of activities in the given population.

    Args:
        population (pandas.DataFrame): A DataFrame containing the population data.

    Returns:
        pandas.Series: A Series containing the participation rate for all pairs of activities.
    """
    act_counts = (
        population.groupby("pid").act.value_counts().unstack(fill_value=0)
    )
    acts = list(population.act.unique())
    pairs = combinations_with_replacement(acts, 2)
    metric = {}
    for pair in pairs:
        counts = calc_pair_rate(act_counts, pair)
        keys = array(list(counts.keys()))
        values = array(list(counts.values()))
        metric["+".join(pair)] = (keys, values / values.sum())

    return metric
