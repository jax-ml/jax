import jax.numpy as jnp
from jax import jit, vmap

# define primitives for distance functions according to scipy.spatial.distance
# which can be batched over the first dimension and then used inside jit and vmap
# as well as used for implementing cdist and pdist versions

primitives = {
    "braycurtis": lambda x, y: jnp.sum(jnp.abs(x - y), axis=-1)
    / jnp.sum(jnp.abs(x + y), axis=-1),
    "canberra": lambda x, y: jnp.sum(
        jnp.abs(x - y) / (jnp.abs(x) + jnp.abs(y)), axis=-1
    ),
    "chebyshev": lambda x, y: jnp.max(jnp.abs(x - y), axis=-1),
    "cityblock": lambda x, y: jnp.sum(jnp.abs(x - y), axis=-1),
    "correlation": lambda x, y: 1
    - jnp.sum((x - y) ** 2, axis=-1)
    / ((jnp.sum(x**2, axis=-1) * jnp.sum(y**2, axis=-1)) ** 0.5),
    "cosine": lambda x, y: 1
    - jnp.sum(x * y, axis=-1)
    / (jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1)),
    "dice": lambda x, y: jnp.sum(x != y, axis=-1)
    / (jnp.sum(x != 0, axis=-1) + jnp.sum(y != 0, axis=-1)),
    "euclidean": jnp.linalg.norm,
    "hamming": lambda x, y: jnp.sum(x != y, axis=-1) / x.shape[-1],
    "jaccard": lambda x, y: jnp.sum(x != y, axis=-1)
    / jnp.sum((x != y) | (x != 0) | (y != 0), axis=-1),
    "kulczynski1": lambda x, y: (jnp.sum(x != y, axis=-1) - jnp.sum(x == y, axis=-1))
    / (jnp.sum(x != y, axis=-1) + jnp.sum(x == y, axis=-1)),
    "mahalanobis": lambda x, y, V: jnp.sqrt(jnp.sum(((x - y) / V) ** 2, axis=-1)),
    "matching": lambda x, y: jnp.sum(x != y, axis=-1) / x.shape[-1],
    "minkowski": lambda x, y, p: jnp.sum(jnp.abs(x - y) ** p, axis=-1) ** (1 / p),
    "rogerstanimoto": lambda x, y: (jnp.sum(x != y, axis=-1) - jnp.sum(x == y, axis=-1))
    / (2 * jnp.sum(x != y, axis=-1) + jnp.sum(x == y, axis=-1)),
    "russellrao": lambda x, y: jnp.sum(x != y, axis=-1) / x.shape[-1],
    "seuclidean": lambda x, y, V: jnp.sqrt(jnp.sum(((x - y) / V) ** 2, axis=-1)),
    "sokalmichener": lambda x, y: (jnp.sum(x != y, axis=-1) - jnp.sum(x == y, axis=-1))
    / (jnp.sum(x != y, axis=-1) + jnp.sum(x == y, axis=-1)),
    "sokalsneath": lambda x, y: (jnp.sum(x != y, axis=-1) - jnp.sum(x == y, axis=-1))
    / (jnp.sum(x != y, axis=-1) + jnp.sum(x == y, axis=-1)),
    "sqeuclidean": lambda x, y: jnp.sum((x - y) ** 2, axis=-1),
    "yule": lambda x, y: (jnp.sum(x != y, axis=-1) - jnp.sum(x == y, axis=-1))
    / (jnp.sum(x != y, axis=-1) + jnp.sum(x == y, axis=-1)),
}

# vectorize primitives for batched, jit'ed evaluation. Take special care for the primitives that
# take additional arguments (p, V)
_METRICS = {}
for k, v in primitives.items():
    if k in ("minkowski", "mahalanobis", "seuclidean"):
        _METRICS[k] = jit(vmap(v, in_axes=(0, 0, None)))
    else:
        _METRICS[k] = jit(vmap(v))


def pdist(X, metric="euclidean", p=2, V=None):
    """
    Pairwise distance between observations in n-dimensional space.

    Parameters
    ----------
    X : ndarray
        An array of shape (n, m) representing n observations in m dimensions.
    metric : str or callable, optional
        The distance metric to use. The distance function can be "braycurtis",
        "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
        "euclidean", "hamming", "jaccard", "kulczynski1", "mahalanobis",
        "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
        "sokalmichener", "sokalsneath", "sqeuclidean", or "yule".
        If a callable is passed it must be a function of the form
        ``f(X, Y) -> ndarray``.
    p : float, optional
        The p-norm to apply for Minkowski, weighted and unweighted
        Euclidean distance. Default is 2.
    V : ndarray, optional
        The variance vector for Mahalanobis distance. Default is None.

    Returns
    -------
    ndarray
        Returns an array of shape (n * (n - 1) / 2, ) representing the upper
        triangle of the distance matrix.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax.scipy.spatial.distance import pdist
    >>> X = jnp.array([[0, 1], [1, 1], [2, 2]])
    >>> pdist(X, metric="euclidean")
    array([1.        , 2.23606798, 1.41421356])
    >>> pdist(X, metric="cityblock")
    array([1., 3., 2.])
    >>> pdist(X, metric="sqeuclidean")
    array([1., 5., 2.])
    >>> pdist(X, metric="hamming")
    array([0.5, 1. , 0.5])
    >>> pdist(X, metric="jaccard")
    array([0.5, 1. , 0.5])
    >>> pdist(X, metric="chebyshev")
    array([1., 2., 1.])
    >>> pdist(X, metric="canberra")
    array([0.66666667, 1.66666667, 1.        ])
    >>> pdist(X, metric="braycurtis")
    array([0.2, 0.5, 0.2])
    >>> pdist(X, metric="mahalanobis", V=jnp.array([1, 1]))
    array([1.        , 1.41421356, 0.70710678])
    >>> pdist(X, metric="minkowski", p=1)
    array([1., 3., 2.])
    >>> pdist(X, metric="minkowski", p=3)
    array([1.        , 2.27950706, 1.25992105])
    >>> pdist(X, metric="seuclidean", V=jnp.array([1, 1]))
    array([1.        , 1.41421356, 0.70710678])
    >>> pdist(X, metric="sqeuclidean")
    array([1., 5., 2.])
    >>> pdist(X, metric="yule")
    array([0.5, 1. , 0.5])
    >>> pdist(X, metric="matching")
    array([0.5, 1. , 0.5])
    >>> pdist(X, metric="dice")
    array([0.33333333, 0.5       , 0.33333333])
    >>> pdist(X, metric="kulszynski1")
    array([0.33333333, 0.5       , 0.33333333])
    >>> pdist(X, metric="rogerstanimoto")
    array([0.33333333, 0.5       , 0.33333333])
    >>> pdist(X, metric="russellrao")
    array([0.5, 1. , 0.5])
    >>> pdist(X, metric="sokalmichener")
    array([0.33333333, 0.5       , 0.33333333])
    >>> pdist(X, metric="sokalsneath")
    array([0.33333333, 0.5       , 0.33333333])
    """
    if metric not in _METRICS:
        raise ValueError(f"Unknown distance metric {metric}.")

    if metric in ("mahalanobis", "seuclidean"):
        if V is None:
            raise ValueError(
                "Variance vector V must be specified for Mahalanobis distance."
            )
        V = jnp.asarray(V)
        if V.ndim != 1:
            raise ValueError("Variance vector V must be one-dimensional.")

    if callable(metric):
        return metric(X)

    if metric == "minkowski":
        return _METRICS[metric](X, p=p)

    return _METRICS[metric](X, X, V=V)
