SQRT_DOC = """Return the non-negative square-root of an array, element-wise.

LAX-backend implementation of :func:`sqrt`.
Original docstring below.

sqrt(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Parameters
----------
x : array_like
    The values whose square-roots are required.

Returns
-------
y : ndarray
    An array of the same shape as `x`, containing the positive
    square-root of each element in `x`.  If any element in `x` is
    complex, a complex array is returned (and the square-roots of
    negative reals are calculated).  If all of the elements in `x`
    are real, so is `y`, with negative elements returning ``nan``.
    If `out` was provided, `y` is a reference to it.
    This is a scalar if `x` is a scalar.

Notes
-----
*sqrt* has--consistent with common convention--as its branch cut the
real "interval" [`-inf`, 0), and is continuous from above on it.
A branch cut is a curve in the complex plane across which a given
complex function fails to be continuous.

Examples
--------
>>> np.sqrt([1,4,9])
array([ 1.,  2.,  3.])

>>> np.sqrt([4, -1, -3+4J])
array([ 2.+0.j,  0.+1.j,  1.+2.j])

>>> np.sqrt([4, -1, np.inf])
array([ 2., nan, inf])
"""


dc_dict = {'sqrt':SQRT_DOC}