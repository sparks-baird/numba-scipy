"""
Nopython version of dist_matrix.

Author: Sterling Baird
"""
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from scipy.spatial.distance import cdist
from time import time
from numba import prange, njit
from numba.types import float32, int32
import numpy as np

np_int = np.int32
np_float = np.float32

nb_int = int32
nb_float = float32

# generate test data
np.random.seed(42)
rows = 200
cols = 100
[U, V, U_weights, V_weights] = [np.random.rand(rows, cols) for i in range(4)]

testpairs = np.array([(1, 2), (2, 3), (3, 4)])

fastmath = True
parallel = True
debug = False

n_neighbors = 3
n_neighbors2 = 1


class Timer(object):
    """
    Simple timer class.

    https://stackoverflow.com/a/5849861/13697228
    Usage
    -----
    with Timer("description"):
        # do stuff
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[%s]" % self.name,)
        print(("Elapsed: {}\n").format(round((time() - self.tstart), 5)))


def setdiff(a, b):
    """
    Find the rows in a which are not in b.

    Source: modified from https://stackoverflow.com/a/11903368/13697228
    See also: https://www.mathworks.com/help/matlab/ref/double.setdiff.html

    Parameters
    ----------
    a : 2D array
        Set of vectors.
    b : 2D array
        Set of vectors.

    Returns
    -------
    out : 2D array
        Set of vectors in a that are not in b.

    """
    a_rows = a.view([("", a.dtype)] * a.shape[1])
    b_rows = b.view([("", b.dtype)] * b.shape[1])
    out = np.setdiff1d(a_rows, b_rows).view(a.dtype).reshape(-1, a.shape[1])
    return out


@njit(fastmath=fastmath, debug=debug)
def wasserstein_distance_check(u_values, v_values, u_weights=None, v_weights=None, p=1):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:
    .. math::
        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}
    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.
    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.
    Returns
    -------
    distance : float
        The computed distance between the distributions.
    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.
    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.
    """
    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind="mergesort")

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], "right")
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], "right")

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(
            ([0], np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(
            ([0], np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    return np.power(
        np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p), deltas)), 1 / p
    )


@njit(fastmath=fastmath, debug=debug)
def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None, p=1, presorted=False):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:
    .. math::
        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}
    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Source: https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L8404

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
            Munos "The Cramer Distance as a Solution to Biased Wasserstein
            Gradients" (2017). :arXiv:`1705.10743`.
    """

    if not presorted:
        u_sorter = np.argsort(u_values)
        v_sorter = np.argsort(v_values)

        u_values = u_values[u_sorter]
        v_values = v_values[v_sorter]

        u_weights = u_weights[u_sorter]
        v_weights = v_weights[v_sorter]

    all_values = np.concatenate((u_values, v_values))
    # all_values.sort(kind='mergesort')
    all_values.sort()

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = np.searchsorted(
        u_values, all_values[:-1], side="right")
    v_cdf_indices = np.searchsorted(
        v_values, all_values[:-1], side="right")

    zero = np.array([0])
    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        uw_cumsum = np.cumsum(u_weights)
        u_sorted_cumweights = np.concatenate((zero, uw_cumsum))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        vw_cumsum = np.cumsum(v_weights)
        v_sorted_cumweights = np.concatenate((zero, vw_cumsum))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.power(u_cdf - v_cdf, 2), deltas)))
    return np.power(
        np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p), deltas)), 1 / p
    )


@njit(fastmath=fastmath, debug=debug)
def euclidean_distance(a, b):
    """
    Calculate Euclidean distance between vectors a and b.

    Parameters
    ----------
    a : 1D array
        First vector.
    b : 1D array
        Second vector.

    Returns
    -------
    d : numeric scalar
        Euclidean distance between vectors a and b.
    """
    d = 0
    for i in range(len(a)):
        d += (b[i] - a[i]) ** 2
    d = np.sqrt(d)
    return d


@njit(fastmath=fastmath, debug=debug)
def compute_distance(u, v, u_weights, v_weights, metric_num):
    """
    Calculate weighted distance between two vectors, u and v.

    Parameters
    ----------
    u : 1D array of float
        First vector.
    v : 1D array of float
        Second vector.
    u_weights : 1D array of float
        Weights for u.
    v_weights : 1D array of float
        Weights for v.
    metric_num : int
        Which metric to use (0 == "euclidean", 1=="wasserstein").

    Raises
    ------
    NotImplementedError
        "Specified metric is mispelled or has not been implemented yet. If not implemented, consider submitting a pull request."

    Returns
    -------
    d : float
        Weighted distance between u and v.

    """
    if metric_num == 0:
        # d = np.linalg.norm(vec - vec2)
        d = euclidean_distance(u, v)
    elif metric_num == 1:
        # d = my_wasserstein_distance(vec, vec2)
        d = wasserstein_distance(
            u, v, u_weights=u_weights, v_weights=v_weights, p=1, presorted=True)
    else:
        raise NotImplementedError(
            "Specified metric is mispelled or has not been implemented yet. If not implemented, consider submitting a pull request."
        )
    return d


@njit(fastmath=fastmath, parallel=parallel, debug=debug)
def sparse_distance_matrix(U, V, U_weights, V_weights, pairs, out, isXY, metric_num):
    """
    Calculate sparse pairwise distances between two sets of vectors for pairs.

    Parameters
    ----------
    mat : numeric cuda array
        First set of vectors for which to compute a single pairwise distance.
    mat2 : numeric cuda array
        Second set of vectors for which to compute a single pairwise distance.
    pairs : cuda array of 2-tuples
        All pairs for which distances are to be computed.
    out : numeric cuda array
        The initialized array which will be populated with distances.

    Raises
    ------
    ValueError
        Both matrices should have the same number of columns.

    Returns
    -------
    None.

    """
    npairs = pairs.shape[0]

    for k in prange(npairs):
        pair = pairs[k]
        i, j = pair

        u = U[i]
        v = V[j]
        uw = U_weights[i]
        vw = V_weights[j]

        d = compute_distance(u, v, uw, vw, metric_num)
        out[k] = d


# @njit(fastmath=fastmath, parallel=parallel, debug=debug)
def one_set_distance_matrix(U, U_weights, out, metric_num):
    """
    Calculate pairwise distances within single set of vectors.

    Parameters
    ----------
    U : 2D array of float
        Vertically stacked vectors.
    U_weights : 2D array of float
        Vertically stacked weight vectors.
    out : 2D array of float
        Initialized matrix to populate with pairwise distances.
    metric_num : int
        Which metric to use (0 == "euclidean", 1=="wasserstein").

    Returns
    -------
    None.

    """
    dm_rows = U.shape[0]
    dm_cols = U.shape[0]

    for i in range(dm_rows):
        for j in range(dm_cols):
            if i < j:
                # if i < j and i < dm_rows and j < dm_cols:
                u = U[i]
                v = U[j]
                uw = U_weights[i]
                vw = U_weights[j]
                d = compute_distance(u, v, uw, vw, metric_num)
                out[i, j] = d
                out[j, i] = d


@njit(fastmath=fastmath, parallel=parallel, debug=debug)
def two_set_distance_matrix(U, V, U_weights, V_weights, out, metric_num):

    # distance matrix shape
    dm_rows = U.shape[0]
    dm_cols = V.shape[0]
    for i in prange(dm_rows):
        for j in range(dm_cols):
            # if i < dm_rows and j < dm_cols:
            u = U[i]
            v = V[j]
            uw = U_weights[i]
            vw = V_weights[j]
            d = compute_distance(u, v, uw, vw, metric_num)
            out[i, j] = d


# @njit(fastmath=fastmath, debug=debug)
def dist_matrix(U, V=None, Uw=None, Vw=None, pairs=None, metric="euclidean"):
    """
    Compute pairwise distances using Numba/CUDA.

    Parameters
    ----------
    mat : array
        First set of vectors for which to compute pairwise distances.

    mat2 : array, optional
        Second set of vectors for which to compute pairwise distances. If not specified,
        then mat2 is a copy of mat.

    pairs : array, optional
        List of 2-tuples which contain the indices for which to compute distances for.
        If mat2 was specified, then the second index accesses mat2 instead of mat.
        If not specified, then the pairs are auto-generated. If mat2 was specified,
        all combinations of the two vector sets are used. If mat2 isn't specified,
        then only the upper triangle (minus diagonal) pairs are computed.

    metric : str, optional
        Possible options are 'euclidean', 'wasserstein'.
        Defaults to Euclidean distance. These are converted to integers internally
        due to Numba's lack of support for string arguments (2021-08-14).
        See compute_distance() for other keys. For example, 0 corresponds to Euclidean
        distance and 1 corresponds to Wasserstein distance.

    Returns
    -------
    out : array
        A pairwise distance matrix, or if pairs are specified, then a vector of
        distances corresponding to the pairs.

    """
    # is it a distance matrix between two sets of vectors rather than within a single set?
    isXY = V is not None

    # were pairs specified? (useful for sparse matrix generation)
    pairQ = pairs is not None

    # assign metric_num based on specified metric (Numba doesn't support strings)
    metric_dict = {"euclidean": 0, "wasserstein": 1}
    metric_num = metric_dict[metric]

    m = U.shape[0]

    if isXY:
        m2 = V.shape[0]
    else:
        m2 = m

    if pairQ:
        npairs = pairs.shape[0]
        shape = (npairs, 1)
    else:
        shape = (m, m2)

    if metric == "wasserstein":
        U_sorter = np.argsort(U)
        U = np.take_along_axis(U, U_sorter, axis=-1)
        Uw = np.take_along_axis(Uw, U_sorter, axis=-1)

        if isXY:
            V_sorter = np.argsort(V)
            V = np.take_along_axis(V, V_sorter, axis=-1)
            Vw = np.take_along_axis(Vw, V_sorter, axis=-1)

    out = np.zeros(shape, dtype=np_float)

    if isXY and not pairQ:
        # distance matrix between two sets of vectors
        two_set_distance_matrix(U, V, Uw, Vw, out, metric_num)

    elif not isXY and pairQ:
        # specified pairwise distances within single set of vectors
        sparse_distance_matrix(U, U, Uw, Uw, pairs, out, isXY, metric_num)

    elif not isXY and not pairQ:
        # distance matrix within single set of vectors
        one_set_distance_matrix(U, Uw, out, metric_num)

    elif isXY and pairQ:
        # specified pairwise distances between two sets of vectors
        sparse_distance_matrix(U, V, Uw, Vw, pairs, out, isXY, metric_num)

    return out


# %% example runs
pairtest = np.array([(0, 1), (1, 2), (2, 3)])
Utest = U[0:6]
Vtest = V[0:6]
Uwtest = U_weights[0:6]
Vwtest = V_weights[0:6]

one_set = dist_matrix(Utest, Uw=Uwtest, metric="wasserstein")
print(one_set)

two_set = dist_matrix(Utest, V=Vtest, Uw=Uwtest,
                      Vw=Vwtest, metric="wasserstein")
print(two_set)

pairs = np.array([(0, 1), (1, 2), (2, 3)])

one_set_sparse = dist_matrix(
    Utest, Uw=Uwtest, pairs=pairtest, metric="wasserstein")
print(one_set_sparse)

two_set_sparse = dist_matrix(
    Utest, V=Vtest, Uw=Uwtest, Vw=Vwtest, pairs=pairtest, metric="wasserstein"
)
print(two_set_sparse)

# %% unit tests
one_set_check = scipy_wasserstein_distance(
    U[0], U[1], u_weights=U_weights[0], v_weights=U_weights[1]
)
two_set_check = scipy_wasserstein_distance(
    U[0], V[0], u_weights=U_weights[0], v_weights=V_weights[0]
)
i, j = pairs[0]
one_sparse_check = scipy_wasserstein_distance(
    U[i], U[j], u_weights=U_weights[i], v_weights=U_weights[j]
)
two_sparse_check = scipy_wasserstein_distance(
    U[i], V[j], u_weights=U_weights[i], v_weights=V_weights[j]
)

tol = 1e-6
print(
    abs(one_set[0, 1] - one_set_check) < tol,
    abs(two_set[0, 0] - two_set_check) < tol,
    abs(one_set_sparse[0, 0] - one_sparse_check) < tol,
    abs(two_set_sparse[0, 0] - two_sparse_check) < tol,
)


# %% wasserstein helper functions
def my_wasserstein_distance(u_uw, v_vw):
    """
    Return Earth Mover's distance using concatenated values and weights.

    Parameters
    ----------
    u_uw : 1D numeric array
        Horizontally stacked values and weights of first distribution.
    v_vw : TYPE
        Horizontally stacked values and weights of second distribution.

    Returns
    -------
    distance : numeric scalar
        Earth Mover's distance given two distributions.

    """
    # split into values and weights
    n = len(u_uw)
    i = n // 2
    u = u_uw[0:i]
    uw = u_uw[i:n]
    v = v_vw[0:i]
    vw = v_vw[i:n]
    # calculate distance
    distance = wasserstein_distance(u, v, u_weights=uw, v_weights=vw)
    return distance


def join_wasserstein(U, V, Uw, Vw):
    """
    Horizontally stack values and weights for each distribution.

    Weights are added as additional columns to values.

    Example:
        u_uw, v_vw = join_wasserstein(u, v, uw, vw)
        d = my_wasserstein_distance(u_uw, v_vw)
        cdist(u_uw, v_vw, metric=my_wasserstein_distance)

    Parameters
    ----------
    u : 1D or 2D numeric array
        First set of distribution values.
    v : 1D or 2D numeric array
        Second set of values of distribution values.
    uw : 1D or 2D numeric array
        Weights for first distribution.
    vw : 1D or 2D numeric array
        Weights for second distribution.

    Returns
    -------
    u_uw : 1D or 2D numeric array
        Horizontally stacked values and weights of first distribution.
    v_vw : TYPE
        Horizontally stacked values and weights of second distribution.

    """
    U_Uw = np.concatenate((U, Uw), axis=1)
    V_Vw = np.concatenate((V, Vw), axis=1)
    return U_Uw, V_Vw


U_Uw, V_Vw = join_wasserstein(U, V, U_weights, V_weights)

# %% timing of large distance matrices


# for compilation purposes, maybe just once is necessary?
dist_matrix(U, Uw=U_weights, metric="wasserstein")
dist_matrix(Utest, V=Vtest, Uw=Uwtest, Vw=Vwtest, metric="wasserstein")
dist_matrix(Utest, Uw=Uwtest, pairs=pairtest, metric="wasserstein")
dist_matrix(Utest, V=Vtest, Uw=Uwtest, Vw=Vwtest,
            pairs=pairtest, metric="wasserstein")

with Timer("cdist Euclidean"):
    d = cdist(U, V)

with Timer("two-set dist_matrix Euclidean"):
    d = dist_matrix(U, V=V, Uw=U_weights, Vw=V_weights, metric="euclidean")

with Timer("cdist SciPy Wasserstein"):
    d = cdist(U_Uw, V_Vw, metric=scipy_wasserstein_distance)

with Timer("cdist Wasserstein"):
    d = cdist(U_Uw, V_Vw, metric=my_wasserstein_distance)

with Timer("two-set dist_matrix Wasserstein"):
    d = dist_matrix(U, V=V, Uw=U_weights, Vw=V_weights, metric="wasserstein")
