"""
Contains the path technology behind opt_einsum in addition to several path helpers
"""
from __future__ import absolute_import, division, print_function

import heapq
import itertools
from collections import defaultdict

import numpy as np

from . import helpers

__all__ = ["optimal", "branch", "greedy"]


_UNLIMITED_MEM = {-1, None, float('inf')}


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


def _calc_k12_flops(inputs, output, remaining, i, j, size_dict):
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

    best = {'flops': float('inf'), 'path': (tuple(range(len(inputs))),)}
    size_cache = {}
    result_cache = {}

    def _optimal_iterate(path, remaining, inputs, flops):

        # reached end of path (only ever get here if flops is best found so far)
        if len(remaining) == 1:
            best['flops'] = flops
            best['path'] = path
            return

        # check all possible remaining paths
        for i, j in itertools.combinations(remaining, 2):
            if i > j:
                i, j = j, i
            key = (inputs[i], inputs[j])
            try:
                k12, flops12 = result_cache[key]
            except KeyError:
                k12, flops12 = result_cache[key] = _calc_k12_flops(inputs, output, remaining, i, j, size_dict)

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
                        best['path'] = path + (tuple(remaining),)
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

    return ssa_to_linear(best['path'])


def branch(inputs, output, size_dict, memory_limit=None, nbranch=None):
    """
    Explores possible pair contractions in a depth-first recursive manner like
    the ``optimal`` approach, but with extra heuristic early pruning of
    branches as well sieving by ``memory_limit`` and the best path found so far.
    Returns the lowest cost path. This algorithm still scales factorially with
    respect to the elements in the list ``input_sets`` if ``nbranch`` is not
    set, but it scales exponentially like ``nbranch**len(input_sets)``
    otherwise.

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
    nbranch : None or int, optional
        How many branches to explore at each contraction step. If None, explore
        all possible branches. If an integer, branch into this many paths at
        each step.

    Returns
    -------
    path : list
        The contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> branch(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    inputs = tuple(map(frozenset, inputs))
    output = frozenset(output)

    best = {'flops': float('inf'), 'path': (tuple(range(len(inputs))),)}
    best_progress = defaultdict(lambda: float('inf'))

    size_cache = {k: helpers.compute_size_by_dict(k, size_dict) for k in inputs}
    result_cache = {}

    def _branch_iterate(path, inputs, remaining, flops):

        # reached end of path (only ever get here if flops is best found so far)
        if len(remaining) == 1:
            best['flops'] = flops
            best['path'] = path
            return

        def _assess_candidate(k1, k2, i, j):
            # find resulting indices and cost
            try:
                k12, flops12 = result_cache[k1, k2]
            except KeyError:
                k12, flops12 = result_cache[k1, k2] = _calc_k12_flops(inputs, output, remaining, i, j, size_dict)

            # sieve based on current best flops
            new_flops = flops + flops12
            if new_flops >= best['flops']:
                return None

            # compare to how the best method was doing as this point
            if new_flops < best_progress[len(inputs)]:
                best_progress[len(inputs)] = new_flops
            # sieve based on current progress relative to best
            elif new_flops > 4 * best_progress[len(inputs)]:
                return None

            # create a sort order based on cost
            size1, size2 = size_cache[inputs[i]], size_cache[inputs[j]]
            try:
                size12 = size_cache[k12]
            except KeyError:
                size12 = size_cache[k12] = helpers.compute_size_by_dict(k12, size_dict)

            # sieve based on memory limit
            if (memory_limit not in _UNLIMITED_MEM) and (size12 > memory_limit):
                # terminate path here, but check all-terms contract first
                new_flops = flops + _compute_oversize_flops(inputs, remaining, output, size_dict)
                if new_flops < best['flops']:
                    best['flops'] = new_flops
                    best['path'] = path + (tuple(remaining),)
                return None

            # set cost heuristic in order to locally sort possible contractions
            cost = size12 - size1 - size2
            return cost, flops12, new_flops, (i, j), k12

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
                k1, k2 = inputs[i], inputs[j]
                candidate = _assess_candidate(k1, k2, i, j)
                if candidate:
                    heapq.heappush(candidates, candidate)

        if not candidates:
            return

        # recurse into all or some of the best candidate contractions
        bi = 0
        while (nbranch is None or bi < nbranch) and candidates:
            _, _, new_flops, (i, j), k12 = heapq.heappop(candidates)
            _branch_iterate(path=path + ((i, j),),
                            inputs=inputs + (k12,),
                            remaining=remaining - {i, j} | {len(inputs)},
                            flops=new_flops)
            bi += 1

    _branch_iterate(path=(),
                    inputs=inputs,
                    remaining=set(range(len(inputs))),
                    flops=0)

    return ssa_to_linear(best['path'])


def _get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2):
    either = k1 | k2
    two = k1 & k2
    one = either - two
    k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
    cost = helpers.compute_size_by_dict(k12, sizes) - footprints[k1] - footprints[k2]
    id1 = remaining[k1]
    id2 = remaining[k2]
    if id1 > id2:
        k1, id1, k2, id2 = k2, id2, k1, id1
    cost = cost, id2, id1  # break ties to ensure determinism
    return cost, k1, k2, k12


def _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue):
    candidate = min(_get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2)
                    for k2 in k2s)
    heapq.heappush(queue, candidate)


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


def _ssa_optimize(inputs, output, sizes):
    """
    This is the core function for :func:`greedy` but produces a path with
    static single assignment ids rather than recycled linear ids.
    SSA ids are cheaper to work with and easier to reason about.
    """
    if len(inputs) == 1:
        # Perform a single contraction to match output shape.
        return [(0,)]

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
            _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue)

    # Greedily contract pairs of tensors.
    while queue:
        cost, k1, k2, k12 = heapq.heappop(queue)
        if k1 not in remaining or k2 not in remaining:
            continue  # candidate is obsolete

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
            _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue)

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


def greedy(inputs, output, size_dict, memory_limit=None):
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
        return branch(inputs, output, size_dict, memory_limit, nbranch=1)

    ssa_path = _ssa_optimize(inputs, output, size_dict)
    return ssa_to_linear(ssa_path)
