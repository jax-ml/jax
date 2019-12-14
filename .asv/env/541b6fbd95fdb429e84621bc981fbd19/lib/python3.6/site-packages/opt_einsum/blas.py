"""
Determines if a contraction can use BLAS or not
"""

import numpy as np

from . import helpers

__all__ = ["can_blas", "tensor_blas"]


def can_blas(inputs, result, idx_removed, shapes=None):
    """
    Checks if we can use a BLAS call.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation
    shapes : sequence of tuple[int], optional
        If given, check also that none of the indices are broadcast dimensions.

    Returns
    -------
    type : str or bool
        The type of BLAS call to be used or False if none.

    Notes
    -----
    We assume several operations are not efficient such as a transposed
    DDOT, therefore 'ijk,jki->' should prefer einsum. These return the blas
    type appended with "/EINSUM" to differentiate when they can still be done
    with tensordot if required, e.g. when a backend has no einsum.

    Examples
    --------
    >>> can_blas(['ij', 'jk'], 'ik', set('j'))
    'GEMM'

    >>> can_blas(['ijj', 'jk'], 'ik', set('j'))
    False

    >>> can_blas(['ab', 'cd'], 'abcd', set())
    'OUTER/EINSUM'

    >>> # looks like GEMM but actually 'j' is broadcast:
    >>> can_blas(['ij', 'jk'], 'ik', set('j'), shapes=[(4, 1), (5, 6)])
    False
    """
    # Can only do two
    if len(inputs) != 2:
        return False

    input_left, input_right = inputs

    for c in set(input_left + input_right):
        # can't deal with repeated indices on same input or more than 2 total
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False

        # can't do implicit summation or dimension collapse e.g.
        #     "ab,bc->c" (implicitly sum over 'a')
        #     "ab,ca->ca" (take diagonal of 'a')
        if nl + nr - 1 == int(c in result):
            return False

    # check for broadcast indices e.g:
    #     "ij,jk->ik" (but one of the 'j' dimensions is broadcast up)
    if shapes is not None:
        for c in idx_removed:
            if shapes[0][input_left.find(c)] != shapes[1][input_right.find(c)]:
                return False

    # Prefer einsum if not removing indices
    #     (N.B. tensordot outer faster for large arrays?)
    if len(idx_removed) == 0:
        return 'OUTER/EINSUM'

    # Build a few temporaries
    sets = [set(x) for x in inputs]
    keep_left = sets[0] - idx_removed
    keep_right = sets[1] - idx_removed
    rs = len(idx_removed)

    # DDOT
    if inputs[0] == inputs[1]:
        return 'DOT'

    # DDOT doesnt make sense if you have to tranpose - prefer einsum
    elif sets[0] == sets[1]:
        return 'DOT/EINSUM'

    # GEMM no transpose
    if input_left[-rs:] == input_right[:rs]:
        return 'GEMM'

    # GEMM transpose both
    elif input_left[:rs] == input_right[-rs:]:
        return 'GEMM'

    # GEMM transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        return 'GEMM'

    # GEMM tranpose left
    elif input_left[:rs] == input_right[:rs]:
        return 'GEMM'

    # Einsum is faster than vectordot if we have to copy
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        return 'GEMV/EINSUM'

    # Conventional tensordot
    else:
        return 'TDOT'


def tensor_blas(view_left, input_left, view_right, input_right, index_result, idx_removed):
    """
    Computes the dot product between two tensors, attempts to use np.dot and
    then tensordot if that fails.

    Parameters
    ----------
    view_left : array_like
        The left hand view
    input_left : str
        Indices of the left view
    view_right : array_like
        The right hand view
    input_right : str
        Indices of the right view
    index_result : str
        The resulting indices
    idx_removed : set
        Indices removed in the contraction

    Returns
    -------
    type : array
        The resulting BLAS operation.

    Notes
    -----
    Interior function for tensor BLAS.

    This function will attempt to use `np.dot` by the iterating through the
    four possible transpose cases. If this fails all inner and matrix-vector
    operations will be handed off to einsum while all matrix-matrix operations will
    first copy the data, perform the DGEMM, and then copy the data to the required
    order.

    Examples
    --------

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4)
    >>> tmp = tensor_blas(a, 'ij', b, 'jk', 'ik', set('j'))
    >>> np.allclose(tmp, np.dot(a, b))

    """

    idx_removed = set(idx_removed)
    keep_left = set(input_left) - idx_removed
    keep_right = set(input_right) - idx_removed

    # We trust this must be called correctly
    dimension_dict = {}
    for i, s in zip(input_left, view_left.shape):
        dimension_dict[i] = s
    for i, s in zip(input_right, view_right.shape):
        dimension_dict[i] = s

    # Do we want to be able to do this?

    # Check for duplicate indices, cannot do einsum('iij,jkk->ik') operations here
    # if (len(set(input_left)) != len(input_left)):
    #     new_inds = ''.join(keep_left) + ''.join(idx_removed)
    #     view_left = np.einsum(input_left + '->' + new_inds, view_left, order='C')
    #     input_left = new_inds

    # if (len(set(input_right)) != len(input_right)):
    #     new_inds = ''.join(idx_removed) + ''.join(keep_right)
    #     view_right = np.einsum(input_right + '->' + new_inds, view_right, order='C')
    #     input_right = new_inds

    # Tensordot guarantees a copy for ndim > 2, should avoid skip if possible
    rs = len(idx_removed)
    dim_left = helpers.compute_size_by_dict(keep_left, dimension_dict)
    dim_right = helpers.compute_size_by_dict(keep_right, dimension_dict)
    dim_removed = helpers.compute_size_by_dict(idx_removed, dimension_dict)
    tensor_result = input_left + input_right
    for s in idx_removed:
        tensor_result = tensor_result.replace(s, "")

    # This is ugly, but can vastly speed up certain operations
    # Vectordot
    if input_left == input_right:
        new_view = np.dot(view_left.ravel(), view_right.ravel())

    # Matrix multiply
    # No transpose needed
    elif input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(view_left.reshape(dim_left, dim_removed), view_right.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(view_left.reshape(dim_removed, dim_left).T, view_right.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(view_left.reshape(dim_left, dim_removed), view_right.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(view_left.reshape(dim_removed, dim_left).T, view_right.reshape(dim_removed, dim_right))

    # Conventional tensordot
    else:
        # Find indices to contract over
        left_pos, right_pos = (), ()
        for s in idx_removed:
            left_pos += (input_left.find(s), )
            right_pos += (input_right.find(s), )
        new_view = np.tensordot(view_left, view_right, axes=(left_pos, right_pos))

    # Make sure the resulting shape is correct
    tensor_shape = tuple(dimension_dict[x] for x in tensor_result)
    if new_view.shape != tensor_shape:
        if len(tensor_result) > 0:
            new_view.shape = tensor_shape
        else:
            new_view = np.squeeze(new_view)

    if tensor_result != index_result:
        new_view = np.einsum(tensor_result + '->' + index_result, new_view)

    return new_view
