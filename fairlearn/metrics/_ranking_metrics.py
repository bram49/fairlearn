# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import random
import math

""""
Metrics for measuring fairness in rankings

Implemented metrics are:
- Normalized discounted difference (rND)
- Normalized discounted KL-divergence (rKL)
    can be used for non-binary protected group memberships
- Normalized discounted ratio (rRD)
    can only be used when protected group is the minority group

source:
Ke Yang, Julia Stoyanovich (2017). Measuring Fairness in Ranked Outputs
https://dl.acm.org/doi/10.1145/3085504.3085526
"""

# Global variables
ND_DIFFERENCE = "rND"  # represent normalized difference group fairness measure
KL_DIVERGENCE = "rKL"  # represent kl-divergence group fairness measure
RD_DIFFERENCE = "rRD"  # represent ratio difference group fairness measure
LOG_BASE = 2  # log base used in logarithm function

NORM_CUTPOINT = 10  # cut-off point used in normalizer computation
NORM_ITERATION = 10  # max iterations used in normalizer computation

_INVALID_GF_MEASURE = "Invalid group fairness measure was passed, " \
                      "please choose one of the following: {'rND', 'rKL', 'rRD'}"


def calculate_normalized_fairness(ranking, protected_group, cut_point,
                                  group_fairness_measure, normalizer):
    """Calculate group fairness value of the whole ranking.

    Calls function 'calculateFairness' in the calculation.

    Parameters
    ----------
    ranking : list, tuple, np.ndarray
        A permutation of N numbers (0..N-1) that represents a ranking of N individuals,
        e.g., [0, 3, 5, 2, 1, 4].  Each number is an identifier of an individual.
    protected_group : list, tuple, np.ndarray
        A set of identifiers from _ranking that represent members of the protected group
        e.g., [0, 2, 3].  Stored as a python array for convenience, order does not matter.
    cut_point : int
        Cut range for the calculation of group fairness, e.g., 10, 20, 30,...
    group_fairness_measure : {'rND', 'rKL', 'rRD'} :
        Group fairness measure to be used in the calculation, one of 'rKL', 'rND', 'rRD'.
        rRD can only be used when protected group is the minority group.
    normalizer : float
        The normalizer of the input group_fairness_measure
        is computed externally for efficiency.

    Returns
    -------
    float
        Fairness value of ranking, normalized to [0, 1]
    """
    # error handling for ranking and protected group
    check_ranking_properties(ranking, protected_group)

    # error handling for input type
    if not isinstance(cut_point, int):
        raise TypeError("Input batch size must be an integer larger than 0")
    if not isinstance(normalizer, (int, float, complex)):
        raise TypeError("Input normalizer must be a number larger than 0")
    if not isinstance(group_fairness_measure, str):
        raise TypeError("Input group fairness measure must be a string. "
                        "Choose from ['rKL', 'rND', 'rRD']")

    user_number = len(ranking)
    protected_number = len(protected_group)

    # error handling for input value
    if NORM_CUTPOINT > user_number:
        raise ValueError("Batch size should be less than input ranking's length")

    discounted_gf = 0  # initialize the returned group fairness value
    for countni in range(user_number):
        countni = countni + 1
        if countni % cut_point == 0:
            ranking_cutpoint = ranking[0:countni]
            protected_cutpoint = set(ranking_cutpoint).intersection(protected_group)

            gf = calculate_fairness(ranking_cutpoint, protected_cutpoint, user_number,
                                    protected_number, group_fairness_measure)
            discounted_gf += gf / math.log(countni + 1, LOG_BASE)  # log base -> global variable

    if normalizer == 0:
        raise ValueError("Normalizer equals to zero")
    return discounted_gf / normalizer


def calculate_fairness(ranking, protected_group, user_number,
                       protected_user_number, group_fairness_measure):
    """Calculate the group fairness value of input ranking.

    Called by function 'calculate_normalized_Fairness'

    Parameters
    ----------
    ranking : list, tuple, np.ndarray
        A permutation of N numbers (0..N-1) that represents a ranking of N individuals,
        e.g., [0, 3, 5, 2, 1, 4].  Each number is an identifier of an individual.
    protected_group : list, tuple, np.ndarray
        A set of identifiers from _ranking that represent members of the protected group
        e.g., [0, 2, 3].  Stored as a python array for convenience, order does not matter.
    user_number : int
        The total user number of input ranking
    protected_user_number : int
        The size of protected group in the input ranking
    group_fairness_measure : {'rND', 'rKL', 'rRD'} :
        Group fairness measure to be used in the calculation, one of 'rKL', 'rND', 'rRD'.
        rRD can only be used when protected group is the minority group.

    Returns
    -------
    float
        returns the value of selected group fairness measure of this input ranking
    """
    k = len(ranking)
    protected_in_k = len(protected_group)
    if group_fairness_measure == KL_DIVERGENCE:  # for KL-divergence difference
        gf = calculator_kl(k, protected_in_k, user_number, protected_user_number)

    elif group_fairness_measure == ND_DIFFERENCE:  # for normalized difference
        gf = calculator_nd(k, protected_in_k, user_number, protected_user_number)

    elif group_fairness_measure == RD_DIFFERENCE:  # for ratio difference
        gf = calculator_rd(k, protected_in_k, user_number, protected_user_number)
    else:
        raise ValueError(_INVALID_GF_MEASURE)

    return gf


def calculator_kl(k, protected_k, user_number, protected_number):
    """Calculate the KL-divergence difference of input ranking.

    Parameters
    ----------
    k : int
        Size of the top-k
    protected_k : int
        Size of protected group in top-k
    user_number : int
        The total size of input items
    protected_number : int
        The total size of input protected group

    Returns
    -------
    float
        The value of KL-divergence difference of this input ranking
    """
    px = protected_k / k
    qx = protected_number / user_number
    if px == 0 or px == 1:  # manually set the value of extreme case to avoid errors
        px = 0.001
    if qx == 0 or qx == 1:
        qx = 0.001
    return px * math.log(px / qx, LOG_BASE) + (1 - px) * math.log((1 - px) / (1 - qx), LOG_BASE)


def calculator_nd(k, protected_k, user_number, protected_number):
    """Calculate the normalized difference of input ranking.

    Parameters
    ----------
    k : int
        Size of the top-k
    protected_k : int
        Size of protected group in top-k
    user_number : int
        The total size of input items
    protected_number : int
        The total size of input protected group

    Returns
    -------
    float
        The value of normalized difference of this input ranking
    """
    return abs(protected_k / k - protected_number / user_number)


def calculator_rd(k, protected_k, user_number, protected_number):
    """Calculate the ratio difference of input ranking.

    Parameters
    ----------
    k : int
        Size of the top-k
    protected_k : int
        Size of protected group in top-k
    user_number : int
        The total size of input items
    protected_number : int
        The total size of input protected group

    Returns
    -------
    float
        The value of ratio difference of this input ranking
    """
    input_ratio = protected_number / (user_number - protected_number)
    unprotected_k = k - protected_k

    if unprotected_k == 0:  # manually set the case of denominator equals zero
        current_ratio = 0
    else:
        current_ratio = protected_k / unprotected_k

    min_ratio = min(input_ratio, current_ratio)

    return abs(min_ratio - input_ratio)


""""
    Methods used to calculate Normalizer Z
        Normalizer Z is computed as the highest possible value of the group fairness metric
        for the given number of items N and protected group size |S^+|.
"""


def calculate_normalizer(user_number, protected_user_number, group_fairness_measure):
    """Calculate the normalizer of input group fairness measure.

    The function use two constant: NORM_ITERATION AND NORM_CUTPOINT to specify the max iteration
    and batch size used in the calculation.
    First, get the maximum value of input group fairness measure at different fairness probability.
    Run the above calculation NORM_ITERATION times.
    Then compute the average value of above results as the maximum value of each fairness
    probability.
    Finally, choose the maximum of value as the normalizer of this group fairness measure.

    Parameters
    ----------
    user_number : int
        The total user number of input ranking
    protected_user_number : int
        The size of protected group in the input ranking
    group_fairness_measure : {'rKL', 'rND', 'rRD'}
        The group fairness measure for the unfair ranking generated at input setting

    Returns
    -------
    float
        The group fairness value for the unfair ranking generated at input setting
    """
    # RD_difference not meaningful when f_prob > 0.5
    f_probs = [0, 0.5] if group_fairness_measure == RD_DIFFERENCE else [0, 1]
    avg_maximums = []  # initialize the lists of average results of all iteration
    for f_prob in f_probs:
        iter_results = []  # initialize the lists of results of all iteration
        for iteri in range(NORM_ITERATION):
            input_ranking = [x for x in range(user_number)]
            protected_group = [x for x in range(protected_user_number)]
            # generate unfair ranking using algorithm
            unfair_ranking = generate_unfair_ranking(input_ranking, protected_group, f_prob)
            # calculate the non-normalized group fairness value i.e. input normalized value as 1
            gf = calculate_normalized_fairness(unfair_ranking, protected_group,
                                               NORM_CUTPOINT, group_fairness_measure, 1)
            iter_results.append(gf)
        avg_maximums.append(np.mean(iter_results))
    return max(avg_maximums)


def generate_unfair_ranking(ranking, protected_group, fairness_probability):
    """Reranks the ranking with given unfairness degree.

    Parameters
    ----------
    ranking : list, tuple, np.ndarray
        A ranking
    protected_group : list, tuple, np.ndarray
        The indices in the ranking of people belonging to the protected group
    fairness_probability : float
        The unfair degree, where 0 is most unfair (unprotected group ranked first)
        and 1 is fair (groups are mixed randomly in the output ranking)

    Returns
    -------
    list
        A ranking that has the specified degree of unfairness w.r.t. the protected group
    """
    # error handling for ranking and protected group
    check_ranking_properties(ranking, protected_group)

    if not isinstance(fairness_probability, (int, float, complex)):
        raise TypeError("Input fairness probability must be a number")
    # error handling for value
    if fairness_probability > 1 or fairness_probability < 0:
        raise ValueError("Input fairness probability must be a number in [0,1]")

    pro_ranking = [x for x in ranking if x not in protected_group]
    unpro_ranking = [x for x in ranking if x in protected_group]
    pro_ranking.reverse()  # prepare for pop function to get the first element
    unpro_ranking.reverse()
    unfair_ranking = []

    while len(unpro_ranking) > 0 and len(pro_ranking) > 0:
        random_seed = random.random()  # generate a random value in range [0,1]
        if random_seed < fairness_probability:
            unfair_ranking.append(unpro_ranking.pop())  # insert protected group first
        else:
            unfair_ranking.append(pro_ranking.pop())  # insert unprotected group first

    if len(unpro_ranking) > 0:  # insert the remain unprotected member
        unpro_ranking.reverse()
        unfair_ranking = unfair_ranking + unpro_ranking
    if len(pro_ranking) > 0:  # insert the remain protected member
        pro_ranking.reverse()
        unfair_ranking = unfair_ranking + pro_ranking

    if len(unfair_ranking) < len(ranking):  # check error for insertation
        print("Error!")
    return unfair_ranking


# Function for error handling
def check_ranking_properties(ranking, protected_group):
    """Check whether input ranking and protected group is valid.

    Parameters
    ----------
    ranking : list, tuple, np.ndarray
        A ranking
    protected_group : list, tuple, np.ndarray
        The indices in the ranking of people belonging to the protected group

    Returns
    -------
        No returns. Raise errors if founded.
    """
    # error handling for input type
    if not isinstance(ranking, (list, tuple, np.ndarray)):
        raise TypeError("Input ranking must be a list-wise structure defined by '[]' symbol")
    if not isinstance(protected_group, (list, tuple, np.ndarray)):
        raise TypeError("Input protected group must be a list-wise structure "
                        "defined by '[]' symbol")

    user_number = len(ranking)
    protected_number = len(protected_group)

    # error handling for input value
    if user_number <= 0:  # check size of input ranking
        raise ValueError("Please input a valid ranking")
    if protected_number <= 0:  # check size of input ranking
        raise ValueError("Please input a valid protected group whose length is larger than 0")

    # check size of protected group
    if protected_number >= user_number:
        raise ValueError("Please input a protected group with size less than total user")
    # check for repetition in input ranking
    if len(set(ranking)) != user_number:
        raise ValueError("Please input a valid complete ranking")
    # check repetition of protected group
    if len(set(protected_group)) != protected_number:
        raise ValueError("Please input a valid protected group that have no repetitive members")
    # check valid of protected group
    if len(set(protected_group).intersection(ranking)) <= 0:
        raise ValueError("Please input a valid protected group that is a subset of total user")
    # check valid of protected group
    if len(set(protected_group).intersection(ranking)) != protected_number:
        raise ValueError("Please input a valid protected group that is a subset of total user")
