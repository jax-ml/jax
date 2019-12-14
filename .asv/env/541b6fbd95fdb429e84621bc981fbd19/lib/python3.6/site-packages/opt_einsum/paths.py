"""
Contains the path technology behind opt_einsum in addition to several path helpers
"""

import functools
import heapq
import itertools
import random
from collections import defaultdict

import numpy as np

from . import helpers

__all__ = [
    "optimal", "BranchBound", "branch", "greedy", "auto", "get_path_fn",
    "DynamicProgrammingOptimizer", "dynamic_programming"
]


_UNLIMITED_MEM = {-1, None, float('inf')}


class PathOptimizer(object):
    """Base class for different path optimizers to inherit from.

    Subclassed optimizers should define a call method with signature::

        def __call__(self, inputs, output, size_dict, memory_limit=None):
            \"\"\"
            Parameters
            ----------
            inputs : list[set[str]]
                The indices of each input array.
            outputs : set[str]
                The output indices
            size_dict : dict[str, int]
                The size of each index
            memory_limit : int, optional
                If given, the maximum allowed memory.
            \"\"\"
            # ... compute path here ...
            return path

    where ``path`` is a list of int-tuples specifiying a contraction order.
    """

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        raise NotImplementedError


def ssa_to_linear(ssa_path):
    """
    Convert a path with static single assignment ids to a path with recycled
    linear ids. For example::

        >>> ssa_to_linear([(0, 3), (2, 4), (1, 5)])
        [(0, 3), (1, 2), (0, 1)]
    """
    ids = np.arange(1 + max(map(max, ssa_path)), dtype=np.int32)
    path = []
    for ssa_ids in ssa_path:
        path.append(tuple(int(ids[ssa_id]) for ssa_id in ssa_ids))
        for ssa_id in ssa_ids:
            ids[ssa_id:] -= 1
    return path


def linear_to_ssa(path):
    """
    Convert a path with recycled linear ids to a path with static single
    assignment ids. For example::

        >>> linear_to_ssa([(0, 3), (1, 2), (0, 1)])
        [(0, 3), (2, 4), (1, 5)]
    """
    num_inputs = sum(map(len, path)) - len(path) + 1
    linear_to_ssa = list(range(num_inputs))
    new_ids = itertools.count(num_inputs)
    ssa_path = []
    for ids in path:
        ssa_path.append(tuple(linear_to_ssa[id_] for id_ in ids))
        for id_ in sorted(ids, reverse=True):
            del linear_to_ssa[id_]
        linear_to_ssa.append(next(new_ids))
    return ssa_path


def calc_k12_flops(inputs, output, remaining, i, j, size_dict):
    """
    Calculate the resulting indices and flops for a potential pairwise
    contraction - used in the recursive (optimal/branch) algorithms.

    Parameters
    ----------
    inputs : tuple[frozenset[str]]
        The indices of each tensor in this contraction, note this includes
        tensors unavaiable to contract as static single assignment is used ->
        contracted tensors are not removed from the list.
    output : frozenset[str]
        The set of output indices for the whole contraction.
    remaining : frozenset[int]
        The set of indices (corresponding to ``inputs``) of tensors still
        available to contract.
    i : int
        Index of potential tensor to contract.
    j : int
        Index of potential tensor to contract.
    size_dict dict[str, int]
        Size mapping of all the indices.

    Returns
    -------
    k12 : frozenset
        The resulting indices of the potential tensor.
    cost : int
        Estimated flop count of operation.
    """
    k1, k2 = inputs[i], inputs[j]
    either = k1 | k2
    shared = k1 & k2
    keep = frozenset.union(output, *map(inputs.__getitem__, remaining - {i, j}))

    k12 = either & keep
    cost = helpers.flop_count(either, shared - keep, 2, size_dict)

    return k12, cost


def _compute_oversize_flops(inputs, remaining, output, size_dict):
    """
    Compute the flop count for a contraction of all remaining arguments. This
    is used when a memory limit means that no pairwise contractions can be made.
    """
    idx_contraction = frozenset.union(*map(inputs.__getitem__, remaining))
    inner = idx_contraction - output
    num_terms = len(remaining)
    return helpers.flop_count(idx_contraction, inner, num_terms, size_dict)


def optimal(inputs, output, size_dict, memory_limit=None):
    """
    Computes all possible pair contractions in a depth-first recursive manner,
    sieving results based on ``memory_limit`` and the best path found so far.
    Returns the lowest cost path. This algorithm scales factoriallly with
    respect to the elements in the list ``input_sets``.

    Parameters
    ----------
    inputs : list
        List of sets that represent the lhs side of the einsum subscript.
    output : set
        Set that represents the rhs side of the overall einsum subscript.
    size_dict : dictionary
        Dictionary of index sizes.
    memory_limit : int
        The maximum number of elements in a temporary array.

    Returns
    -------
    path : list
        The optimal contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> optimal(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """
    inputs = tuple(map(frozenset, inputs))
    output = frozenset(output)

    best = {'flops': float('inf'), 'ssa_path': (tuple(range(len(inputs))),)}
    size_cache = {}
    result_cache = {}

    def _optimal_iterate(path, remaining, inputs, flops):

        # reached end of path (only ever get here if flops is best found so far)
        if len(remaining) == 1:
            best['flops'] = flops
            best['ssa_path'] = path
            return

        # check all possible remaining paths
        for i, j in itertools.combinations(remaining, 2):
            if i > j:
                i, j = j, i
            key = (inputs[i], inputs[j])
            try:
                k12, flops12 = result_cache[key]
            except KeyError:
                k12, flops12 = result_cache[key] = calc_k12_flops(inputs, output, remaining, i, j, size_dict)

            # sieve based on current best flops
            new_flops = flops + flops12
            if new_flops >= best['flops']:
                continue

            # sieve based on memory limit
            if memory_limit not in _UNLIMITED_MEM:
                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = helpers.compute_size_by_dict(k12, size_dict)

                # possibly terminate this path with an all-terms einsum
                if size12 > memory_limit:
                    new_flops = flops + _compute_oversize_flops(inputs, remaining, output, size_dict)
                    if new_flops < best['flops']:
                        best['flops'] = new_flops
                        best['ssa_path'] = path + (tuple(remaining),)
                    continue

            # add contraction and recurse into all remaining
            _optimal_iterate(path=path + ((i, j),),
                             inputs=inputs + (k12,),
                             remaining=remaining - {i, j} | {len(inputs)},
                             flops=new_flops)

    _optimal_iterate(path=(),
                     inputs=inputs,
                     remaining=set(range(len(inputs))),
                     flops=0)

    return ssa_to_linear(best['ssa_path'])


# functions for comparing which of two paths is 'better'

def better_flops_first(flops, size, best_flops, best_size):
    return (flops, size) < (best_flops, best_size)


def better_size_first(flops, size, best_flops, best_size):
    return (size, flops) < (best_size, best_flops)


_BETTER_FNS = {
    'flops': better_flops_first,
    'size': better_size_first,
}


def get_better_fn(key):
    return _BETTER_FNS[key]


# functions for assigning a heuristic 'cost' to a potential contraction

def cost_memory_removed(size12, size1, size2, k12, k1, k2):
    """The default heuristic cost, corresponding to the total reduction in
    memory of performing a contraction.
    """
    return size12 - size1 - size2


def cost_memory_removed_jitter(size12, size1, size2, k12, k1, k2):
    """Like memory-removed, but with a slight amount of noise that breaks ties
    and thus jumbles the contractions a bit.
    """
    return random.gauss(1.0, 0.01) * (size12 - size1 - size2)


_COST_FNS = {
    'memory-removed': cost_memory_removed,
    'memory-removed-jitter': cost_memory_removed_jitter,
}


class BranchBound(PathOptimizer):
    """
    Explores possible pair contractions in a depth-first recursive manner like
    the ``optimal`` approach, but with extra heuristic early pruning of branches
    as well sieving by ``memory_limit`` and the best path found so far. Returns
    the lowest cost path. This algorithm still scales factorially with respect
    to the elements in the list ``input_sets`` if ``nbranch`` is not set, but it
    scales exponentially like ``nbranch**len(input_sets)`` otherwise.

    Parameters
    ----------
    nbranch : None or int, optional
        How many branches to explore at each contraction step. If None, explore
        all possible branches. If an integer, branch into this many paths at
        each step. Defaults to None.
    cutoff_flops_factor : float, optional
        If at any point, a path is doing this much worse than the best path
        found so far was, terminate it. The larger this is made, the more paths
        will be fully explored and the slower the algorithm. Defaults to 4.
    minimize : {'flops', 'size'}, optional
        Whether to optimize the path with regard primarily to the total
        estimated flop-count, or the size of the largest intermediate. The
        option not chosen will still be used as a secondary criterion.
    cost_fn : callable, optional
        A function that returns a heuristic 'cost' of a potential contraction
        with which to sort candidates. Should have signature
        ``cost_fn(size12, size1, size2, k12, k1, k2)``.
    """

    def __init__(self, nbranch=None, cutoff_flops_factor=4, minimize='flops', cost_fn='memory-removed'):
        self.nbranch = nbranch
        self.cutoff_flops_factor = cutoff_flops_factor
        self.minimize = minimize
        self.cost_fn = _COST_FNS.get(cost_fn, cost_fn)

        self.better = get_better_fn(minimize)
        self.best = {'flops': float('inf'), 'size': float('inf')}
        self.best_progress = defaultdict(lambda: float('inf'))

    @property
    def path(self):
        return ssa_to_linear(self.best['ssa_path'])

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        """

        Parameters
        ----------
        input_sets : list
            List of sets that represent the lhs side of the einsum subscript
        output_set : set
            Set that represents the rhs side of the overall einsum subscript
        idx_dict : dictionary
            Dictionary of index sizes
        memory_limit : int
            The maximum number of elements in a temporary array

        Returns
        -------
        path : list
            The contraction order within the memory limit constraint.

        Examples
        --------
        >>> isets = [set('abd'), set('ac'), set('bdc')]
        >>> oset = set('')
        >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
        >>> optimal(isets, oset, idx_sizes, 5000)
        [(0, 2), (0, 1)]
        """

        inputs = tuple(map(frozenset, inputs))
        output = frozenset(output)

        size_cache = {k: helpers.compute_size_by_dict(k, size_dict) for k in inputs}
        result_cache = {}

        def _branch_iterate(path, inputs, remaining, flops, size):

            # reached end of path (only ever get here if flops is best found so far)
            if len(remaining) == 1:
                self.best['size'] = size
                self.best['flops'] = flops
                self.best['ssa_path'] = path
                return

            def _assess_candidate(k1, k2, i, j):
                # find resulting indices and flops
                try:
                    k12, flops12 = result_cache[k1, k2]
                except KeyError:
                    k12, flops12 = result_cache[k1, k2] = calc_k12_flops(inputs, output, remaining, i, j, size_dict)

                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = helpers.compute_size_by_dict(k12, size_dict)

                new_flops = flops + flops12
                new_size = max(size, size12)

                # sieve based on current best i.e. check flops and size still better
                if not self.better(new_flops, new_size, self.best['flops'], self.best['size']):
                    return None

                # compare to how the best method was doing as this point
                if new_flops < self.best_progress[len(inputs)]:
                    self.best_progress[len(inputs)] = new_flops
                # sieve based on current progress relative to best
                elif new_flops > self.cutoff_flops_factor * self.best_progress[len(inputs)]:
                    return None

                # sieve based on memory limit
                if (memory_limit not in _UNLIMITED_MEM) and (size12 > memory_limit):
                    # terminate path here, but check all-terms contract first
                    new_flops = flops + _compute_oversize_flops(inputs, remaining, output, size_dict)
                    if new_flops < self.best['flops']:
                        self.best['flops'] = new_flops
                        self.best['ssa_path'] = path + (tuple(remaining),)
                    return None

                # set cost heuristic in order to locally sort possible contractions
                size1, size2 = size_cache[inputs[i]], size_cache[inputs[j]]
                cost = self.cost_fn(size12, size1, size2, k12, k1, k2)

                return cost, flops12, new_flops, new_size, (i, j), k12

            # check all possible remaining paths
            candidates = []
            for i, j in itertools.combinations(remaining, 2):
                if i > j:
                    i, j = j, i
                k1, k2 = inputs[i], inputs[j]

                # initially ignore outer products
                if k1.isdisjoint(k2):
                    continue

                candidate = _assess_candidate(k1, k2, i, j)
                if candidate:
                    heapq.heappush(candidates, candidate)

            # assess outer products if nothing left
            if not candidates:
                for i, j in itertools.combinations(remaining, 2):
                    if i > j:
                        i, j = j, i
                    k1, k2 = inputs[i], inputs[j]
                    candidate = _assess_candidate(k1, k2, i, j)
                    if candidate:
                        heapq.heappush(candidates, candidate)

            # recurse into all or some of the best candidate contractions
            bi = 0
            while (self.nbranch is None or bi < self.nbranch) and candidates:
                _, _, new_flops, new_size, (i, j), k12 = heapq.heappop(candidates)
                _branch_iterate(path=path + ((i, j),),
                                inputs=inputs + (k12,),
                                remaining=(remaining - {i, j}) | {len(inputs)},
                                flops=new_flops,
                                size=new_size)
                bi += 1

        _branch_iterate(path=(),
                        inputs=inputs,
                        remaining=set(range(len(inputs))),
                        flops=0,
                        size=0)

        return self.path


def branch(inputs, output, size_dict, memory_limit=None, **optimizer_kwargs):
    optimizer = BranchBound(**optimizer_kwargs)
    return optimizer(inputs, output, size_dict, memory_limit)


branch_all = functools.partial(branch, nbranch=None)
branch_2 = functools.partial(branch, nbranch=2)
branch_1 = functools.partial(branch, nbranch=1)


def _get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2, cost_fn):
    either = k1 | k2
    two = k1 & k2
    one = either - two
    k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
    cost = cost_fn(helpers.compute_size_by_dict(k12, sizes), footprints[k1], footprints[k2], k12, k1, k2)
    id1 = remaining[k1]
    id2 = remaining[k2]
    if id1 > id2:
        k1, id1, k2, id2 = k2, id2, k1, id1
    cost = cost, id2, id1  # break ties to ensure determinism
    return cost, k1, k2, k12


def _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue, push_all, cost_fn):
    candidates = (_get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2, cost_fn) for k2 in k2s)
    if push_all:
        # want to do this if we e.g. are using a custom 'choose_fn'
        for candidate in candidates:
            heapq.heappush(queue, candidate)
    else:
        heapq.heappush(queue, min(candidates))


def _update_ref_counts(dim_to_keys, dim_ref_counts, dims):
    for dim in dims:
        count = len(dim_to_keys[dim])
        if count <= 1:
            dim_ref_counts[2].discard(dim)
            dim_ref_counts[3].discard(dim)
        elif count == 2:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].discard(dim)
        else:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].add(dim)


def _simple_chooser(queue, remaining):
    """Default contraction chooser that simply takes the minimum cost option.
    """
    cost, k1, k2, k12 = heapq.heappop(queue)
    if k1 not in remaining or k2 not in remaining:
        return None  # candidate is obsolete
    return cost, k1, k2, k12


def ssa_greedy_optimize(inputs, output, sizes, choose_fn=None, cost_fn='memory-removed'):
    """
    This is the core function for :func:`greedy` but produces a path with
    static single assignment ids rather than recycled linear ids.
    SSA ids are cheaper to work with and easier to reason about.
    """
    if len(inputs) == 1:
        # Perform a single contraction to match output shape.
        return [(0,)]

    # set the function that assigns a heuristic cost to a possible contraction
    cost_fn = _COST_FNS.get(cost_fn, cost_fn)

    # set the function that chooses which contraction to take
    if choose_fn is None:
        choose_fn = _simple_chooser
        push_all = False
    else:
        # assume chooser wants access to all possible contractions
        push_all = True

    # A dim that is common to all tensors might as well be an output dim, since it
    # cannot be contracted until the final step. This avoids an expensive all-pairs
    # comparison to search for possible contractions at each step, leading to speedup
    # in many practical problems where all tensors share a common batch dimension.
    inputs = list(map(frozenset, inputs))
    output = frozenset(output) | frozenset.intersection(*inputs)

    # Deduplicate shapes by eagerly computing Hadamard products.
    remaining = {}  # key -> ssa_id
    ssa_ids = itertools.count(len(inputs))
    ssa_path = []
    for ssa_id, key in enumerate(inputs):
        if key in remaining:
            ssa_path.append((remaining[key], ssa_id))
            remaining[key] = next(ssa_ids)
        else:
            remaining[key] = ssa_id

    # Keep track of possible contraction dims.
    dim_to_keys = defaultdict(set)
    for key in remaining:
        for dim in key - output:
            dim_to_keys[dim].add(key)

    # Keep track of the number of tensors using each dim; when the dim is no longer
    # used it can be contracted. Since we specialize to binary ops, we only care about
    # ref counts of >=2 or >=3.
    dim_ref_counts = {
        count: set(dim for dim, keys in dim_to_keys.items() if len(keys) >= count) - output
        for count in [2, 3]}

    # Compute separable part of the objective function for contractions.
    footprints = {key: helpers.compute_size_by_dict(key, sizes) for key in remaining}

    # Find initial candidate contractions.
    queue = []
    for dim, keys in dim_to_keys.items():
        keys = sorted(keys, key=remaining.__getitem__)
        for i, k1 in enumerate(keys[:-1]):
            k2s = keys[1 + i:]
            _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue, push_all, cost_fn)

    # Greedily contract pairs of tensors.
    while queue:

        con = choose_fn(queue, remaining)
        if con is None:
            continue  # allow choose_fn to flag all candidates obsolete
        cost, k1, k2, k12 = con

        ssa_id1 = remaining.pop(k1)
        ssa_id2 = remaining.pop(k2)
        for dim in k1 - output:
            dim_to_keys[dim].remove(k1)
        for dim in k2 - output:
            dim_to_keys[dim].remove(k2)
        ssa_path.append((ssa_id1, ssa_id2))
        if k12 in remaining:
            ssa_path.append((remaining[k12], next(ssa_ids)))
        else:
            for dim in k12 - output:
                dim_to_keys[dim].add(k12)
        remaining[k12] = next(ssa_ids)
        _update_ref_counts(dim_to_keys, dim_ref_counts, k1 | k2 - output)
        footprints[k12] = helpers.compute_size_by_dict(k12, sizes)

        # Find new candidate contractions.
        k1 = k12
        k2s = set(k2 for dim in k1 for k2 in dim_to_keys[dim])
        k2s.discard(k1)
        if k2s:
            _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue, push_all, cost_fn)

    # Greedily compute pairwise outer products.
    queue = [(helpers.compute_size_by_dict(key & output, sizes), ssa_id, key)
             for key, ssa_id in remaining.items()]
    heapq.heapify(queue)
    _, ssa_id1, k1 = heapq.heappop(queue)
    while queue:
        _, ssa_id2, k2 = heapq.heappop(queue)
        ssa_path.append((min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)))
        k12 = (k1 | k2) & output
        cost = helpers.compute_size_by_dict(k12, sizes)
        ssa_id12 = next(ssa_ids)
        _, ssa_id1, k1 = heapq.heappushpop(queue, (cost, ssa_id12, k12))

    return ssa_path


def greedy(inputs, output, size_dict, memory_limit=None, choose_fn=None, cost_fn='memory-removed'):
    """
    Finds the path by a three stage algorithm:

    1. Eagerly compute Hadamard products.
    2. Greedily compute contractions to maximize ``removed_size``
    3. Greedily compute outer products.

    This algorithm scales quadratically with respect to the
    maximum number of elements sharing a common dim.

    Parameters
    ----------
    inputs : list
        List of sets that represent the lhs side of the einsum subscript
    output : set
        Set that represents the rhs side of the overall einsum subscript
    size_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array
    choose_fn : callable, optional
        A function that chooses which contraction to perform from the queu
    cost_fn : callable, optional
        A function that assigns a potential contraction a cost.

    Returns
    -------
    path : list
        The contraction order (a list of tuples of ints).

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> greedy(isets, oset, idx_sizes)
    [(0, 2), (0, 1)]
    """
    if memory_limit not in _UNLIMITED_MEM:
        return branch(inputs, output, size_dict, memory_limit, nbranch=1, cost_fn=cost_fn)

    ssa_path = ssa_greedy_optimize(inputs, output, size_dict, cost_fn=cost_fn, choose_fn=choose_fn)
    return ssa_to_linear(ssa_path)


def _tree_to_sequence(c):
    """
    Converts a contraction tree to a contraction path as it has to be
    returned by path optimizers. A contraction tree can either be an int
    (=no contraction) or a tuple containing the terms to be contracted. An
    arbitrary number (>= 1) of terms can be contracted at once. Note that
    contractions are commutative, e.g. (j, k, l) = (k, l, j). Note that in
    general, solutions are not unique.

    Parameters
    ----------
    c : tuple or int
        Contraction tree

    Returns
    -------
    path : list[set[int]]
        Contraction path

    Examples
    --------
    >>> _tree_to_sequence(((1,2),(0,(4,5,3))))
    [(1, 2), (1, 2, 3), (0, 2), (0, 1)]
    """

    # ((1,2),(0,(4,5,3))) --> [(1, 2), (1, 2, 3), (0, 2), (0, 1)]
    #
    # 0     0         0           (1,2)       --> ((1,2),(0,(3,4,5)))
    # 1     3         (1,2)   --> (0,(3,4,5))
    # 2 --> 4     --> (3,4,5)
    # 3     5
    # 4     (1,2)
    # 5
    #
    # this function iterates through the table shown above from right to left;

    if type(c) == int:
        return []

    c = [c]  # list of remaining contractions (lower part of columns shown above)
    t = []   # list of elementary tensors (upper part of colums)
    s = []   # resulting contraction sequence

    while len(c) > 0:
        j = c.pop(-1)
        s.insert(0, tuple())

        for i in sorted([i for i in j if type(i) == int]):
            s[0] += (sum(1 for q in t if q < i),)
            t.insert(s[0][-1], i)

        for i in [i for i in j if type(i) != int]:
            s[0] += (len(t) + len(c),)
            c.append(i)

    return s


def _find_disconnected_subgraphs(inputs, output):
    """
    Finds disconnected subgraphs in the given list of inputs. Inputs are
    connected if they share summation indices. Note: Disconnected subgraphs
    can be contracted independently before forming outer products.

    Parameters
    ----------
    inputs : list[set]
        List of sets that represent the lhs side of the einsum subscript
    output : set
        Set that represents the rhs side of the overall einsum subscript

    Returns
    -------
    subgraphs : list[set[int]]
        List containing sets of indices for each subgraph

    Examples
    --------
    >>> _find_disconnected_subgraphs([set("ab"), set("c"), set("ad")], set("bd"))
    [{0, 2}, {1}]

    >>> _find_disconnected_subgraphs([set("ab"), set("c"), set("ad")], set("abd"))
    [{0}, {1}, {2}]
    """

    subgraphs = []
    unused_inputs = set(range(len(inputs)))

    i_sum = set.union(*inputs) - output  # all summation indices

    while len(unused_inputs) > 0:
        g = set()
        q = [unused_inputs.pop()]
        while len(q) > 0:
            j = q.pop()
            g.add(j)
            i_tmp = i_sum & inputs[j]
            n = {k for k in unused_inputs if len(i_tmp & inputs[k]) > 0}
            q.extend(n)
            unused_inputs.difference_update(n)

        subgraphs.append(g)

    return subgraphs


def _bitmapset_indices(s):
    """
    Returns a generator object allowing to iterate over the elements contained
    in a bitmap set.

    Parameters
    ----------
    s : int
        The bitmap set to iterate over

    Returns
    -------
    path : generator
        Generator object to iterate over the elements in s

    Examples
    --------
    >>> type(_bitmapset_indices(0b1001011))
    generator

    >>> list(_bitmapset_indices(0b1001011))
    [0, 1, 3, 6]
    """
    j = 0
    while s != 0:
        if s & 1 != 0:
            yield j
        s >>= 1
        j += 1


class DynamicProgrammingOptimizer(PathOptimizer):
    """
    Finds the optimal path of pairwise contractions without intermediate outer
    products based a dynamic programming approach presented in
    Phys. Rev. E 90, 033315 (2014) (the corresponding preprint is publically
    available at https://arxiv.org/abs/1304.6112). This method is especially
    well-suited in the area of tensor network states, where it usually
    outperforms all the other optimization strategies.

    This algorithm shows exponential scaling with the number of inputs
    in the worst case scenario (see example below). If the graph to be
    contracted consists of disconnected subgraphs, the algorithm scales
    linearly in the number of disconnected subgraphs and only exponentially
    with the number of inputs per subgraph.
    """

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        """
        Parameters
        ----------
        inputs : list
            List of sets that represent the lhs side of the einsum subscript
        output : set
            Set that represents the rhs side of the overall einsum subscript
        size_dict : dictionary
            Dictionary of index sizes
        memory_limit : int
            The maximum number of elements in a temporary array

        Returns
        -------
        path : list
            The contraction order (a list of tuples of ints).

        Examples
        --------
        >>> n_in = 3  # exponential scaling
        >>> n_out = 2 # linear scaling
        >>> s = dict()
        >>> i_all = []
        >>> for _ in range(n_out):
        >>>     i = [set() for _ in range(n_in)]
        >>>     for j in range(n_in):
        >>>         for k in range(j+1, n_in):
        >>>             c = oe.get_symbol(len(s))
        >>>             i[j].add(c)
        >>>             i[k].add(c)
        >>>             s[c] = 2
        >>>     i_all.extend(i)
        >>> o = DynamicProgrammingOptimizer()
        >>> o(i_all, set(), s)
        [(1, 2), (0, 4), (1, 2), (0, 2), (0, 1)]
        """

        # convert all indices to integers (makes set operations ~10 % faster)
        symbol2int = {c: j for j, c in enumerate(set.union(*inputs) | output)}
        inputs = [set(symbol2int[c] for c in i) for i in inputs]
        output = set(symbol2int[c] for c in output)
        size_dict = {symbol2int[c]: v for c, v in size_dict.items() if c in symbol2int}
        size_dict = [size_dict[j] for j in range(len(size_dict))]

        # all summation indices occurring exactly in one input:
        i_single = set(
            c for c in set.union(*inputs) - output
            if sum(1 for i in inputs if c in i) == 1
        )

        # contraction expressions for all inputs that have already been
        # reduced to scalars:
        inputs_done = [
            (j,) for j, i in enumerate(inputs)
            if len(i - i_single) == 0
        ]

        # remaining input index sets and corresponding contraction expressions;
        # indices from i_single are removed and if a single-tensor contraction
        # is performed, the contraction expression is (j,) instead of j;
        inputs, inputs_contractions = zip(*[
            (i - i_single, j if i.isdisjoint(i_single) else (j,))
            for j, i in enumerate(inputs)
            if len(i - i_single) > 0
        ])

        # a list of all neccessary contraction expressions for each of the
        # disconnected subgraphs and their size
        subgraph_contractions = inputs_done
        subgraph_contractions_size = [1]*len(inputs_done)

        for g in _find_disconnected_subgraphs(inputs, output):

            # dynamic programming approach to compute x[n] for subgraph g;
            # x[n][set of n tensors] = (indices, cost, contraction)
            # the set of n tensors is represented by a bitmap: if bit j is 1,
            # tensor j is in the set, e.g. 0b100101 = {0,2,5}; set unions
            # (intersections) can then be computed by bitwise or (and);
            x = [None]*2 + [dict() for j in range(len(g)-1)]
            x[1] = {1 << j: (inputs[j], 0, inputs_contractions[j]) for j in g}

            # convert set of tensors g to a bitmap set:
            g = functools.reduce(lambda x, y: x | y, (1 << j for j in g))

            # the bitmap set of all tensors is computed as it is needed to
            # compute set differences: s1 - s2 transforms into
            # s1 & (all_tensors ^ s2)
            all_tensors = (1 << len(inputs)) - 1

            # try to find contraction with cost <= cost_cap and increase
            # cost_cap successively if no such contraction is found;
            # this is a major performance improvement; start with product of
            # output index dimensions as initial cost_cap
            cost_cap = helpers.compute_size_by_dict(
                set.union(*(inputs[j] for j in _bitmapset_indices(g))) & output,
                size_dict
            )

            while len(x[-1]) == 0:
                for n in range(2, len(x[1]) + 1):
                    xn = x[n]

                    # try to combine solutions from x[m] and x[n-m]
                    for m in range(1, n // 2 + 1):
                        for s1, (i1, cost1, cntrct1) in x[m].items():
                            for s2, (i2, cost2, cntrct2) in x[n-m].items():

                                # only if s1 and s2 are disjoint
                                if s1 & s2 == 0:

                                    # avoid e.g. s1={0}, s2={1} and s1={1}, s2={0}
                                    if m != n - m or s1 < s2:

                                        i1_cut_i2_wo_output = (i1 & i2) - output

                                        # ignore outer products:
                                        if len(i1_cut_i2_wo_output) > 0:

                                            i1_union_i2 = i1 | i2
                                            cost = cost1 + cost2 + helpers.compute_size_by_dict(i1_union_i2, size_dict)
                                            if cost <= cost_cap:
                                                s = s1 | s2
                                                if s not in xn or cost < xn[s][1]:
                                                    # set of remaining tensors (=g-s)
                                                    r = g & (all_tensors ^ s)

                                                    # indices of remaining indices:
                                                    i_r = (set.union(*(inputs[j] for j in _bitmapset_indices(r)))
                                                           if r != 0 else set())

                                                    # contraction indices:
                                                    i_cntrct = i1_cut_i2_wo_output - i_r

                                                    i = i1_union_i2 - i_cntrct
                                                    mem = helpers.compute_size_by_dict(i, size_dict)
                                                    if memory_limit is None or mem <= memory_limit:
                                                        xn[s] = (i, cost, (cntrct1, cntrct2))

                # increase cost cap for next iteration:
                cost_cap = min(size_dict) * cost_cap

            i, cost, contraction = list(x[-1].values())[0]
            subgraph_contractions.append(contraction)
            subgraph_contractions_size.append(helpers.compute_size_by_dict(i, size_dict))

        # sort the subgraph contractions by the size of the subgraphs in
        # ascending order (will give the cheapest contractions); note that
        # outer products should be performed pairwise (to use BLAS functions)
        subgraph_contractions = [
            subgraph_contractions[j]
            for j in np.argsort(subgraph_contractions_size)
        ]

        # build the final contraction tree
        tree = functools.reduce(lambda x, y: (x, y), subgraph_contractions)

        return _tree_to_sequence(tree)


def dynamic_programming(inputs, output, size_dict, memory_limit=None):
    optimizer = DynamicProgrammingOptimizer()
    return optimizer(inputs, output, size_dict, memory_limit)


_AUTO_CHOICES = {}
for i in range(1, 5):
    _AUTO_CHOICES[i] = optimal
for i in range(5, 7):
    _AUTO_CHOICES[i] = branch_all
for i in range(7, 9):
    _AUTO_CHOICES[i] = branch_2
for i in range(9, 15):
    _AUTO_CHOICES[i] = branch_1


def auto(inputs, output, size_dict, memory_limit=None):
    """Finds the contraction path by automatically choosing the method based on
    how many input arguments there are.
    """
    N = len(inputs)
    return _AUTO_CHOICES.get(N, greedy)(inputs, output, size_dict, memory_limit)


_PATH_OPTIONS = {
    'auto': auto,
    'optimal': optimal,
    'branch-all': branch_all,
    'branch-2': branch_2,
    'branch-1': branch_1,
    'greedy': greedy,
    'eager': greedy,
    'opportunistic': greedy,
    'dp': dynamic_programming,
    'dynamic-programming': dynamic_programming
}


def register_path_fn(name, fn):
    """Add path finding function ``fn`` as an option with ``name``.
    """
    if name in _PATH_OPTIONS:
        raise KeyError("Path optimizer '{}' already exists.".format(name))

    _PATH_OPTIONS[name.lower()] = fn


def get_path_fn(path_type):
    """Get the correct path finding function from str ``path_type``.
    """
    if path_type not in _PATH_OPTIONS:
        raise KeyError("Path optimizer '{}' not found, valid options are {}."
                       .format(path_type, set(_PATH_OPTIONS.keys())))

    return _PATH_OPTIONS[path_type]
