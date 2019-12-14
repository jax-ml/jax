"""
A top-level linear programming interface. Currently this interface solves
linear programming problems via the Simplex and Interior-Point methods.

.. versionadded:: 0.15.0

Functions
---------
.. autosummary::
   :toctree: generated/

    linprog
    linprog_verbose_callback
    linprog_terse_callback

"""

from __future__ import division, print_function, absolute_import

import numpy as np

from .optimize import OptimizeResult
from ._linprog_ip import _linprog_ip
from ._linprog_simplex import _linprog_simplex
from ._linprog_util import (
    _parse_linprog, _presolve, _get_Abc, _postprocess
    )

__all__ = ['linprog', 'linprog_verbose_callback', 'linprog_terse_callback']

__docformat__ = "restructuredtext en"


def linprog_verbose_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces detailed output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization::

                 0 : Optimization terminated successfully
                 1 : Iteration limit reached
                 2 : Problem appears to be infeasible
                 3 : Problem appears to be unbounded
                 4 : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    x = res['x']
    fun = res['fun']
    success = res['success']
    phase = res['phase']
    status = res['status']
    nit = res['nit']
    message = res['message']
    complete = res['complete']

    saved_printoptions = np.get_printoptions()
    np.set_printoptions(linewidth=500,
                        formatter={'float': lambda x: "{0: 12.4f}".format(x)})
    if status:
        print('--------- Simplex Early Exit -------\n'.format(nit))
        print('The simplex method exited early with status {0:d}'.format(status))
        print(message)
    elif complete:
        print('--------- Simplex Complete --------\n')
        print('Iterations required: {}'.format(nit))
    else:
        print('--------- Iteration {0:d}  ---------\n'.format(nit))

    if nit > 0:
        if phase == 1:
            print('Current Pseudo-Objective Value:')
        else:
            print('Current Objective Value:')
        print('f = ', fun)
        print()
        print('Current Solution Vector:')
        print('x = ', x)
        print()

    np.set_printoptions(**saved_printoptions)


def linprog_terse_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces brief output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization::

                 0 : Optimization terminated successfully
                 1 : Iteration limit reached
                 2 : Problem appears to be infeasible
                 3 : Problem appears to be unbounded
                 4 : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    nit = res['nit']
    x = res['x']

    if nit == 0:
        print("Iter:   X:")
    print("{0: <5d}   ".format(nit), end="")
    print(x)


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
            bounds=None, method='simplex', callback=None,
            options=None):
    """
    Minimize a linear objective function subject to linear
    equality and inequality constraints. Linear Programming is intended to
    solve the following problem form:

    Minimize::

        c @ x

    Subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub

    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.

    Parameters
    ----------
    c : 1D array
        Coefficients of the linear objective function to be minimized.
    A_ub : 2D array, optional
        2D array such that ``A_ub @ x`` gives the values of the upper-bound
        inequality constraints at ``x``.
    b_ub : 1D array, optional
        1D array of values representing the upper-bound of each inequality
        constraint (row) in ``A_ub``.
    A_eq : 2D, optional
        2D array such that ``A_eq @ x`` gives the values of the equality
        constraints at ``x``.
    b_eq : 1D array, optional
        1D array of values representing the RHS of each equality constraint
        (row) in ``A_eq``.
    bounds : sequence, optional
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for one of ``min`` or
        ``max`` when there is no bound in that direction. By default
        bounds are ``(0, None)`` (non-negative).
        If a sequence containing a single tuple is provided, then ``min`` and
        ``max`` will be applied to all variables in the problem.
    method : str, optional
        Type of solver.  :ref:`'simplex' <optimize.linprog-simplex>`
        and :ref:`'interior-point' <optimize.linprog-interior-point>`
        are supported.
    callback : callable, optional (simplex only)
        If a callback function is provided, it will be called within each
        iteration of the simplex algorithm. The callback must require a
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1D array
                The independent variable vector which optimizes the linear
                programming problem.
            fun : float
                Value of the objective function.
            success : bool
                True if the algorithm succeeded in finding an optimal solution.
            slack : 1D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1D array
                The (nominally zero) residuals of the equality constraints
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the optimization being executed. In phase 1 a basic
                feasible solution is sought and the T has an additional row
                representing an alternate objective function.
            status : int
                An integer representing the exit status of the optimization::

                     0 : Optimization terminated successfully
                     1 : Iteration limit reached
                     2 : Problem appears to be infeasible
                     3 : Problem appears to be unbounded
                     4 : Serious numerical difficulties encountered

            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options('linprog')`.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields:

            x : 1D array
                The independent variable vector which optimizes the linear
                programming problem.
            fun : float
                Value of the objective function.
            slack : 1D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                then the corresponding constraint is active.
            con : 1D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            success : bool
                Returns True if the algorithm succeeded in finding an optimal
                solution.
            status : int
                An integer representing the exit status of the optimization::

                     0 : Optimization terminated successfully
                     1 : Iteration limit reached
                     2 : Problem appears to be infeasible
                     3 : Problem appears to be unbounded
                     4 : Serious numerical difficulties encountered

            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.

    See Also
    --------
    show_options : Additional options accepted by the solvers

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method
    is :ref:`Simplex <optimize.linprog-simplex>`.
    :ref:`Interior point <optimize.linprog-interior-point>` is also available.

    Method *simplex* uses the simplex algorithm (as it relates to linear
    programming, NOT the Nelder-Mead simplex) [1]_, [2]_. This algorithm
    should be reasonably reliable and fast for small problems.

    .. versionadded:: 0.15.0

    Method *interior-point* uses the primal-dual path following algorithm
    as outlined in [4]_. This algorithm is intended to provide a faster
    and more reliable alternative to *simplex*, especially for large,
    sparse problems. Note, however, that the solution returned may be slightly
    less accurate than that of the simplex method and may not correspond with a
    vertex of the polytope defined by the constraints.

    Before applying either method a presolve procedure based on [8]_ attempts to
    identify trivial infeasibilities, trivial unboundedness, and potential
    problem simplifications. Specifically, it checks for:

    - rows of zeros in ``A_eq`` or ``A_ub``, representing trivial constraints;
    - columns of zeros in ``A_eq`` `and` ``A_ub``, representing unconstrained
      variables;
    - column singletons in ``A_eq``, representing fixed variables; and
    - column singletons in ``A_ub``, representing simple bounds.

    If presolve reveals that the problem is unbounded (e.g. an unconstrained
    and unbounded variable has negative cost) or infeasible (e.g. a row of
    zeros in ``A_eq`` corresponds with a nonzero in ``b_eq``), the solver
    terminates with the appropriate status code. Note that presolve terminates
    as soon as any sign of unboundedness is detected; consequently, a problem
    may be reported as unbounded when in reality the problem is infeasible
    (but infeasibility has not been detected yet). Therefore, if the output
    message states that unboundedness is detected in presolve and it is
    necessary to know whether the problem is actually infeasible, set option
    ``presolve=False``.

    If neither infeasibility nor unboundedness are detected in a single pass
    of the presolve check, bounds are tightened where possible and fixed
    variables are removed from the problem. Then, linearly dependent rows
    of the ``A_eq`` matrix are removed, (unless they represent an
    infeasibility) to avoid numerical difficulties in the primary solve
    routine. Note that rows that are nearly linearly dependent (within a
    prescribed tolerance) may also be removed, which can change the optimal
    solution in rare cases. If this is a concern, eliminate redundancy from
    your problem formulation and run with option ``rr=False`` or
    ``presolve=False``.

    Several potential improvements can be made here: additional presolve
    checks outlined in [8]_ should be implemented, the presolve routine should
    be run multiple times (until no further simplifications can be made), and
    more of the efficiency improvements from [5]_ should be implemented in the
    redundancy removal routines.

    After presolve, the problem is transformed to standard form by converting
    the (tightened) simple bounds to upper bound constraints, introducing
    non-negative slack variables for inequality constraints, and expressing
    unbounded variables as the difference between two non-negative variables.

    References
    ----------
    .. [1] Dantzig, George B., Linear programming and extensions. Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963
    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
           Mathematical Programming", McGraw-Hill, Chapter 4.
    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
           Mathematics of Operations Research (2), 1977: pp. 103-107.
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [5] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    .. [7] Fourer, Robert. "Solving Linear Programs by Interior-Point Methods."
           Unpublished Course Notes, August 26, 2005. Available 2/25/2017 at
           http://www.4er.org/CourseNotes/Book%20B/B-III.pdf
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [10] Andersen, Erling D., et al. Implementation of interior point
            methods for large scale linear programming. HEC/Universite de
            Geneve, 1996.

    Examples
    --------
    Consider the following problem:

    Minimize::

        f = -1x[0] + 4x[1]

    Subject to::

        -3x[0] + 1x[1] <= 6
         1x[0] + 2x[1] <= 4
                  x[1] >= -3
          -inf <= x[0] <= inf

    This problem deviates from the standard linear programming problem.
    In standard form, linear programming problems assume the variables x are
    non-negative. Since the problem variables don't have the standard bounds of
    ``(0, None)``, the variable bounds must be set using ``bounds`` explicitly.

    There are two upper-bound constraints, which can be expressed as

    dot(A_ub, x) <= b_ub

    The input for this problem is as follows:

    >>> c = [-1, 4]
    >>> A = [[-3, 1], [1, 2]]
    >>> b = [6, 4]
    >>> x0_bounds = (None, None)
    >>> x1_bounds = (-3, None)
    >>> from scipy.optimize import linprog
    >>> res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),
    ...               options={"disp": True})
    Optimization terminated successfully.
    Current function value: -22.000000
    Iterations: 5 # may vary
    >>> print(res)
         con: array([], dtype=float64)
         fun: -22.0
     message: 'Optimization terminated successfully.'
         nit: 5 # may vary
       slack: array([39.,  0.]) # may vary
      status: 0
     success: True
           x: array([10., -3.])

    """
    meth = method.lower()
    default_tol = 1e-12 if meth == 'simplex' else 1e-9

    c, A_ub, b_ub, A_eq, b_eq, bounds, solver_options = _parse_linprog(
        c, A_ub, b_ub, A_eq, b_eq, bounds, options)
    tol = solver_options.get('tol', default_tol)

    iteration = 0
    complete = False    # will become True if solved in presolve
    undo = []

    # Keep the original arrays to calculate slack/residuals for original
    # problem.
    c_o, A_ub_o, b_ub_o, A_eq_o, b_eq_o = c.copy(
    ), A_ub.copy(), b_ub.copy(), A_eq.copy(), b_eq.copy()

    # Solve trivial problem, eliminate variables, tighten bounds, etc...
    c0 = 0  # we might get a constant term in the objective
    if solver_options.pop('presolve', True):
        rr = solver_options.pop('rr', True)
        (c, c0, A_ub, b_ub, A_eq, b_eq, bounds, x, undo, complete, status,
            message) = _presolve(c, A_ub, b_ub, A_eq, b_eq, bounds, rr, tol)

    if not complete:
        A, b, c, c0 = _get_Abc(c, c0, A_ub, b_ub, A_eq, b_eq, bounds, undo)
        T_o = (c_o, A_ub_o, b_ub_o, A_eq_o, b_eq_o, bounds, undo)
        if meth == 'simplex':
            x, status, message, iteration = _linprog_simplex(
                c, c0=c0, A=A, b=b, callback=callback, _T_o=T_o, **solver_options)
        elif meth == 'interior-point':
            x, status, message, iteration = _linprog_ip(
                c, c0=c0, A=A, b=b, callback=callback, **solver_options)
        else:
            raise ValueError('Unknown solver %s' % method)

    # Eliminate artificial variables, re-introduce presolved variables, etc...
    # need modified bounds here to translate variables appropriately
    disp = solver_options.get('disp', False)
    x, fun, slack, con, status, message = _postprocess(
        x, c_o, A_ub_o, b_ub_o, A_eq_o, b_eq_o, bounds,
        complete, undo, status, message, tol, iteration, disp)

    sol = {
        'x': x,
        'fun': fun,
        'slack': slack,
        'con': con,
        'status': status,
        'message': message,
        'nit': iteration,
        'success': status == 0}

    return OptimizeResult(sol)
