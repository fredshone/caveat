from numpy import array, ndarray
from pandas import DataFrame

from caveat.features.utils import weighted_features

# def participation_prob_by_act(population: dict[str, float]) -> Series:
#     """
#     Calculate the participation probability by activity for a given population.

#     Parameters:
#     population (DataFrame): The population data containing the 'pid' and 'act' columns.

#     Returns:
#     Dict: A dictionary object containing the participation probability for each activity.
#     """

#     metrics = population.groupby(["pid", "act"], observed=False).size() > 0
#     metrics = metrics.groupby("act", observed=False).sum()
#     metrics = metrics / population.pid.nunique()
#     metrics = metrics.sort_values(ascending=False)
#     return metrics.to_dict()


def participation_prob_by_act(population: DataFrame) -> dict[str, ndarray]:
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


# def activity_participation_prob(population: DataFrame) -> Series:
#     """
#     Returns a Series containing the probability that an agents participate in each activity.

#     Args:
#         population (DataFrame): A pandas DataFrame containing information about participants and their activities.

#     Returns:
#         Series: A pandas Series where the index is activity names and the values are participation probabilities.
#     """
#     metrics = population.groupby(["pid", "act"], observed=False).size() > 0
#     metrics = (
#         metrics.groupby("act", observed=False)
#         .value_counts()
#         .unstack()
#         .fillna(0)
#     )
#     metrics = metrics.sort_values(ascending=False)
#     return metrics.to_dict(orient="list")


def participation_rates(population: DataFrame) -> dict[str, ndarray]:
    rates = population.groupby("pid").act.count()
    return weighted_features({"all": rates.to_list()})


def participation_rates_by_act(population: DataFrame) -> dict[str, list]:
    rates = population.groupby("pid").act.value_counts().unstack().fillna(0)
    return weighted_features(rates.to_dict(orient="list"))


def participation_rates_by_seq_act(population: DataFrame) -> dict[str, list]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    rates = actseq.groupby(population.pid).value_counts().unstack().fillna(0)
    return weighted_features(rates.to_dict(orient="list"))


def participation_rates_by_act_enum(population: DataFrame) -> dict[str, list]:
    act_enum = population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)
    rates = act_enum.groupby(population.pid).value_counts().unstack().fillna(0)
    return weighted_features(rates.to_dict(orient="list"))


# def participation_rates(population: DataFrame) -> Series:
#     """
#     Calculates the participation rates for each activity in the given population DataFrame.

#     Args:
#         population (DataFrame): A DataFrame containing the population data.

#     Returns:
#         Series: A Series containing the participation rates for each activity.
#     """
#     metrics = (
#         population.groupby("act", observed=False).pid.count()
#         / population.pid.nunique()
#     )
#     metrics.index = MultiIndex.from_tuples(
#         [("participation rate", act) for act in metrics.index]
#     )
#     metrics = metrics.sort_values(ascending=False)
#     return metrics


# def act_plan_seq_participation_rates(population: DataFrame) -> Series:
#     """
#     Calculates the participation rates for each activity (indexed by sequence enumeration) in the given population DataFrame.

#     Args:
#         population (DataFrame): A DataFrame containing the population data.

#     Returns:
#         Series: A Series containing the participation rates for each activity.
#     """
#     actseq = population.act.astype(str) + population.groupby(
#         "pid", as_index=False
#     ).cumcount().astype(str)
#     metrics = population.groupby(actseq).pid.count() / population.pid.nunique()
#     metrics.index = MultiIndex.from_tuples(
#         [("act plan seq participation rate", act) for act in metrics.index]
#     )
#     metrics = metrics.sort_values(ascending=False)
#     return metrics


# def act_seq_participation_rates(population: DataFrame) -> Series:
#     """
#     Calculates the participation rates for each activity (indexed by enumeration) in the given population DataFrame.

#     Args:
#         population (DataFrame): A DataFrame containing the population data.

#     Returns:
#         Series: A Series containing the participation rates for each activity.
#     """
#     actseq = population.act.astype(str) + population.groupby(
#         ["pid", "act"], as_index=False, observed=False
#     ).cumcount().astype(str)
#     metrics = population.groupby(actseq).pid.count() / population.pid.nunique()
#     metrics.index = MultiIndex.from_tuples(
#         [("act seq participation rate", act) for act in metrics.index]
#     )
#     metrics = metrics.sort_values(ascending=False)
#     return metrics


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


def calc_pair_rate(act_counts: DataFrame, pair: tuple) -> float:
    """
    Calculates the participation rate for given activity pairs given activity counts.

    Parameters:
    act_counts (DataFrame): DataFrame of activity counts.
    pair (tuple): Pair of activities to calculate participation rate for.

    Returns:
    float: Participation rate of the pair of users.
    """
    a, b = pair
    if a == b:
        return (act_counts[a] > 1).mean()
    return ((act_counts[a] > 0) & (act_counts[b] > 0)).mean()


def calc_pair_count(act_counts, pair):
    a, b = pair
    if a == b:
        return (act_counts[a] > 1).sum()
    return ((act_counts[a] > 0) & (act_counts[b] > 0)).sum()


def joint_participation_prob(population: DataFrame) -> dict[str, tuple]:
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
    pairs = combinations_with_replacement(list(population.act.unique()), 2)
    n = population.pid.nunique()
    metric = {}
    for pair in pairs:
        p = calc_pair_count(act_counts, pair)
        metric["+".join(pair)] = (array([0, 1]), array([n - p, p]))

    return metric
