"""
Unit test for Linear Programming
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
                           assert_array_less)
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy._lib._numpy_compat import _assert_warns, suppress_warnings
from scipy.sparse.linalg import MatrixRankWarning

import pytest


def magic_square(n):
    np.random.seed(0)
    M = n * (n**2 + 1) / 2

    numbers = np.arange(n**4) // n**2 + 1

    numbers = numbers.reshape(n**2, n, n)

    zeros = np.zeros((n**2, n, n))

    A_list = []
    b_list = []

    # Rule 1: use every number exactly once
    for i in range(n**2):
        A_row = zeros.copy()
        A_row[i, :, :] = 1
        A_list.append(A_row.flatten())
        b_list.append(1)

    # Rule 2: Only one number per square
    for i in range(n):
        for j in range(n):
            A_row = zeros.copy()
            A_row[:, i, j] = 1
            A_list.append(A_row.flatten())
            b_list.append(1)

    # Rule 3: sum of rows is M
    for i in range(n):
        A_row = zeros.copy()
        A_row[:, i, :] = numbers[:, i, :]
        A_list.append(A_row.flatten())
        b_list.append(M)

    # Rule 4: sum of columns is M
    for i in range(n):
        A_row = zeros.copy()
        A_row[:, :, i] = numbers[:, :, i]
        A_list.append(A_row.flatten())
        b_list.append(M)

    # Rule 5: sum of diagonals is M
    A_row = zeros.copy()
    A_row[:, range(n), range(n)] = numbers[:, range(n), range(n)]
    A_list.append(A_row.flatten())
    b_list.append(M)
    A_row = zeros.copy()
    A_row[:, range(n), range(-1, -n - 1, -1)] = \
        numbers[:, range(n), range(-1, -n - 1, -1)]
    A_list.append(A_row.flatten())
    b_list.append(M)

    A = np.array(np.vstack(A_list), dtype=float)
    b = np.array(b_list, dtype=float)
    c = np.random.rand(A.shape[1])

    return A, b, c, numbers


def lpgen_2d(m, n):
    """ -> A b c LP test: m*n vars, m+n constraints
        row sums == n/m, col sums == 1
        https://gist.github.com/denis-bz/8647461
    """
    np.random.seed(0)
    c = - np.random.exponential(size=(m, n))
    Arow = np.zeros((m, m * n))
    brow = np.zeros(m)
    for j in range(m):
        j1 = j + 1
        Arow[j, j * n:j1 * n] = 1
        brow[j] = n / m

    Acol = np.zeros((n, m * n))
    bcol = np.zeros(n)
    for j in range(n):
        j1 = j + 1
        Acol[j, j::n] = 1
        bcol[j] = 1

    A = np.vstack((Arow, Acol))
    b = np.hstack((brow, bcol))

    return A, b, c.ravel()


def _assert_iteration_limit_reached(res, maxiter):
    assert_(not res.success, "Incorrectly reported success")
    assert_(res.success < maxiter, "Incorrectly reported number of iterations")
    assert_equal(res.status, 1, "Failed to report iteration limit reached")


def _assert_infeasible(res):
    # res: linprog result object
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 2, "failed to report infeasible status")


def _assert_unbounded(res):
    # res: linprog result object
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 3, "failed to report unbounded status")


def _assert_unable_to_find_basic_feasible_sol(res):
    # res: linprog result object
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 2, "failed to report optimization failure")


def _assert_success(res, desired_fun=None, desired_x=None,
                    rtol=1e-8, atol=1e-8):
    # res: linprog result object
    # desired_fun: desired objective function value or None
    # desired_x: desired solution or None
    if not res.success:
        msg = "linprog status {0}, message: {1}".format(res.status,
                                                        res.message)
        raise AssertionError(msg)

    assert_equal(res.status, 0)
    if desired_fun is not None:
        assert_allclose(res.fun, desired_fun,
                        err_msg="converged to an unexpected objective value",
                        rtol=rtol, atol=atol)
    if desired_x is not None:
        assert_allclose(res.x, desired_x,
                        err_msg="converged to an unexpected solution",
                        rtol=rtol, atol=atol)


class LinprogCommonTests(object):

    def test_docstring_example(self):
        # Example from linprog docstring.
        c = [-1, 4]
        A = [[-3, 1], [1, 2]]
        b = [6, 4]
        x0_bounds = (None, None)
        x1_bounds = (-3, None)

        res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),
                      options=self.options, method=self.method)
        _assert_success(res, desired_fun=-22)

    def test_aliasing_b_ub(self):
        c = np.array([1.0])
        A_ub = np.array([[1.0]])
        b_ub_orig = np.array([3.0])
        b_ub = b_ub_orig.copy()
        bounds = (-4.0, np.inf)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=-4, desired_x=[-4])
        assert_allclose(b_ub_orig, b_ub)

    def test_aliasing_b_eq(self):
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq_orig = np.array([3.0])
        b_eq = b_eq_orig.copy()
        bounds = (-4.0, np.inf)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=3, desired_x=[3])
        assert_allclose(b_eq_orig, b_eq)

    def test_bounds_second_form_unbounded_below(self):
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq = np.array([3.0])
        bounds = (None, 10.0)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=3, desired_x=[3])

    def test_bounds_second_form_unbounded_above(self):
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq = np.array([3.0])
        bounds = (1.0, None)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=3, desired_x=[3])

    def test_non_ndarray_args(self):
        c = [1.0]
        A_ub = [[1.0]]
        b_ub = [3.0]
        A_eq = [[1.0]]
        b_eq = [2.0]
        bounds = (-1.0, 10.0)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=2, desired_x=[2])

    def test_linprog_upper_bound_constraints(self):
        # Maximize a linear function subject to only linear upper bound
        # constraints.
        #  http://www.dam.brown.edu/people/huiwang/classes/am121/Archive/simplex_121_c.pdf
        c = np.array([3, 2]) * -1  # maximize
        A_ub = [[2, 1],
                [1, 1],
                [1, 0]]
        b_ub = [10, 8, 4]
        res = (linprog(c, A_ub=A_ub, b_ub=b_ub,
                       method=self.method, options=self.options))
        _assert_success(res, desired_fun=-18, desired_x=[2, 6])

    def test_linprog_mixed_constraints(self):
        # Minimize linear function subject to non-negative variables.
        # http://www.statslab.cam.ac.uk/~ff271/teaching/opt/notes/notes8.pdf
        # (dead link)
        c = [6, 3]
        A_ub = [[0, 3],
                [-1, -1],
                [-2, 1]]
        b_ub = [2, -1, -1]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=5, desired_x=[2 / 3, 1 / 3])

    def test_linprog_cyclic_recovery(self):
        # Test linprogs recovery from cycling using the Klee-Minty problem
        #  Klee-Minty  https://www.math.ubc.ca/~israel/m340/kleemin3.pdf
        c = np.array([100, 10, 1]) * -1  # maximize
        A_ub = [[1, 0, 0],
                [20, 1, 0],
                [200, 20, 1]]
        b_ub = [1, 100, 10000]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[0, 0, 10000], atol=5e-6, rtol=1e-7)

    def test_linprog_cyclic_bland(self):
        # Test the effect of Bland's rule on a cycling problem
        c = np.array([-10, 57, 9, 24.])
        A_ub = np.array([[0.5, -5.5, -2.5, 9],
                         [0.5, -1.5, -0.5, 1],
                         [1, 0, 0, 0]])
        b_ub = [0, 0, 1]

        maxiter = 100
        o = {key: val for key, val in self.options.items()}
        o['maxiter'] = maxiter

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, options=o,
                      method=self.method)

        if self.method == 'simplex'and not self.options.get('bland'):
            _assert_iteration_limit_reached(res, o['maxiter'])
        else:
            _assert_success(res, desired_x=[1, 0, 1, 0])

    def test_linprog_cyclic_bland_bug_8561(self):
        # Test that pivot row is chosen correctly when using Bland's rule
        c = np.array([7, 0, -4, 1.5, 1.5])
        A_ub = np.array([
            [4, 5.5, 1.5, 1.0, -3.5],
            [1, -2.5, -2, 2.5, 0.5],
            [3, -0.5, 4, -12.5, -7],
            [-1, 4.5, 2, -3.5, -2],
            [5.5, 2, -4.5, -1, 9.5]])
        b_ub = np.array([0, 0, 0, 0, 1])
        if self.method == "simplex":
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          options=dict(maxiter=100, bland=True),
                          method=self.method)
        else:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, options=self.options,
                          method=self.method)
        _assert_success(res, desired_x=[0, 0, 19, 16/3, 29/3])

    def test_linprog_unbounded(self):
        # Test linprog response to an unbounded problem
        c = np.array([1, 1]) * -1  # maximize
        A_ub = [[-1, 1],
                [-1, -1]]
        b_ub = [-1, -2]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      method=self.method, options=self.options)
        _assert_unbounded(res)

    def test_linprog_infeasible(self):
        # Test linrpog response to an infeasible problem
        c = [-1, -1]
        A_ub = [[1, 0],
                [0, 1],
                [-1, -1]]
        b_ub = [2, 2, -5]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_nontrivial_problem(self):
        # Test linprog for a problem involving all constraint types,
        # negative resource limits, and rounding issues.
        c = [-1, 8, 4, -6]
        A_ub = [[-7, -7, 6, 9],
                [1, -1, -3, 0],
                [10, -10, -7, 7],
                [6, -1, 3, 4]]
        b_ub = [-3, 6, -6, 6]
        A_eq = [[-10, 1, 1, -8]]
        b_eq = [-4]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=7083 / 1391,
                        desired_x=[101 / 1391, 1462 / 1391, 0, 752 / 1391])

    def test_negative_variable(self):
        # Test linprog with a problem with one unbounded variable and
        # another with a negative lower bound.
        c = np.array([-1, 4]) * -1  # maximize
        A_ub = np.array([[-3, 1],
                         [1, 2]], dtype=np.float64)
        A_ub_orig = A_ub.copy()
        b_ub = [6, 4]
        x0_bounds = (-np.inf, np.inf)
        x1_bounds = (-3, np.inf)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(x0_bounds, x1_bounds),
                      method=self.method, options=self.options)

        assert_equal(A_ub, A_ub_orig)   # user input not overwritten
        _assert_success(res, desired_fun=-80 / 7, desired_x=[-8 / 7, 18 / 7])

    def test_large_problem(self):
        # Test linprog simplex with a rather large problem (400 variables,
        # 40 constraints) generated by https://gist.github.com/denis-bz/8647461
        A, b, c = lpgen_2d(20, 20)
        res = linprog(c, A_ub=A, b_ub=b,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=-64.049494229)

    def test_network_flow(self):
        # A network flow problem with supply and demand at nodes
        # and with costs along directed edges.
        # https://www.princeton.edu/~rvdb/542/lectures/lec10.pdf
        c = [2, 4, 9, 11, 4, 3, 8, 7, 0, 15, 16, 18]
        n, p = -1, 1
        A_eq = [
            [n, n, p, 0, p, 0, 0, 0, 0, p, 0, 0],
            [p, 0, 0, p, 0, p, 0, 0, 0, 0, 0, 0],
            [0, 0, n, n, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, p, p, 0, 0, p, 0],
            [0, 0, 0, 0, n, n, n, 0, p, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, n, n, 0, 0, p],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, n, n, n]]
        b_eq = [0, 19, -16, 33, 0, 0, -36]
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=755, atol=1e-6, rtol=1e-7)

    def test_network_flow_limited_capacity(self):
        # A network flow problem with supply and demand at nodes
        # and with costs and capacities along directed edges.
        # http://blog.sommer-forst.de/2013/04/10/
        cost = [2, 2, 1, 3, 1]
        bounds = [
            [0, 4],
            [0, 2],
            [0, 2],
            [0, 3],
            [0, 5]]
        n, p = -1, 1
        A_eq = [
            [n, n, 0, 0, 0],
            [p, 0, n, n, 0],
            [0, p, p, 0, n],
            [0, 0, 0, p, p]]
        b_eq = [-4, 0, 0, 4]

        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "scipy.linalg.solve\nIll...")
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            sup.filter(OptimizeWarning, "Solving system with option...")
            res = linprog(c=cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                          method=self.method, options=self.options)
        _assert_success(res, desired_fun=14)

    def test_simplex_algorithm_wikipedia_example(self):
        # https://en.wikipedia.org/wiki/Simplex_algorithm#Example
        Z = [-2, -3, -4]
        A_ub = [
            [3, 2, 1],
            [2, 5, 3]]
        b_ub = [10, 15]
        res = linprog(c=Z, A_ub=A_ub, b_ub=b_ub,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=-20)

    def test_enzo_example(self):
        # https://github.com/scipy/scipy/issues/1779 lp2.py
        #
        # Translated from Octave code at:
        # http://www.ecs.shimane-u.ac.jp/~kyoshida/lpeng.htm
        # and placed under MIT licence by Enzo Michelangeli
        # with permission explicitly granted by the original author,
        # Prof. Kazunobu Yoshida
        c = [4, 8, 3, 0, 0, 0]
        A_eq = [
            [2, 5, 3, -1, 0, 0],
            [3, 2.5, 8, 0, -1, 0],
            [8, 10, 4, 0, 0, -1]]
        b_eq = [185, 155, 600]
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=317.5,
                        desired_x=[66.25, 0, 17.5, 0, 183.75, 0],
                        atol=6e-6, rtol=1e-7)

    def test_enzo_example_b(self):
        # rescued from https://github.com/scipy/scipy/pull/218
        c = [2.8, 6.3, 10.8, -2.8, -6.3, -10.8]
        A_eq = [[-1, -1, -1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1]]
        b_eq = [-0.5, 0.4, 0.3, 0.3, 0.3]

        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            res = linprog(c=c, A_eq=A_eq, b_eq=b_eq,
                              method=self.method, options=self.options)
        _assert_success(res, desired_fun=-1.77,
                        desired_x=[0.3, 0.2, 0.0, 0.0, 0.1, 0.3])

    def test_enzo_example_c_with_degeneracy(self):
        # rescued from https://github.com/scipy/scipy/pull/218
        m = 20
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(1, m + 1) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = [0, 0]
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=0, desired_x=np.zeros(m))

    def test_enzo_example_c_with_unboundedness(self):
        # rescued from https://github.com/scipy/scipy/pull/218
        m = 50
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(m) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = [0, 0]
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)
        _assert_unbounded(res)

    def test_enzo_example_c_with_infeasibility(self):
        # rescued from https://github.com/scipy/scipy/pull/218
        m = 50
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(m) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = [1, 1]
        o = {key: self.options[key] for key in self.options}
        o["presolve"] = False

        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, method=self.method,
                      options=o)
        _assert_infeasible(res)

    def test_unknown_options(self):
        c = np.array([-3, -2])
        A_ub = [[2, 1], [1, 1], [1, 0]]
        b_ub = [10, 8, 4]

        def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
              options={}):
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method,
                    options=options)

        o = {key: self.options[key] for key in self.options}
        o['spam'] = 42
        _assert_warns(OptimizeWarning, f,
                      c, A_ub=A_ub, b_ub=b_ub, options=o)

    def test_no_constraints(self):
        res = linprog([-1, -2], method=self.method, options=self.options)
        _assert_unbounded(res)

    def test_simple_bounds(self):
        res = linprog([1, 2], bounds=(1, 2),
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[1, 1])
        res = linprog([1, 2], bounds=[(1, 2), (1, 2)],
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[1, 1])

    def test_invalid_inputs(self):

        def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                    method=self.method, options=self.options)

        for bad_bound in [[(5, 0), (1, 2), (3, 4)],
                          [(1, 2), (3, 4)],
                          [(1, 2), (3, 4), (3, 4, 5)],
                          [(1, 2), (np.inf, np.inf), (3, 4)],
                          [(1, 2), (-np.inf, -np.inf), (3, 4)],
                          ]:
            assert_raises(ValueError, f, [1, 2, 3], bounds=bad_bound)

        assert_raises(ValueError, f, [1, 2], A_ub=[[1, 2]], b_ub=[1, 2])
        assert_raises(ValueError, f, [1, 2], A_ub=[[1]], b_ub=[1])
        assert_raises(ValueError, f, [1, 2], A_eq=[[1, 2]], b_eq=[1, 2])
        assert_raises(ValueError, f, [1, 2], A_eq=[[1]], b_eq=[1])
        assert_raises(ValueError, f, [1, 2], A_eq=[1], b_eq=1)

        if ("_sparse_presolve" in self.options and
                self.options["_sparse_presolve"]):
            return
            # this test doesn't make sense for sparse presolve
            # there aren't 3D sparse matrices
        assert_raises(ValueError, f, [1, 2], A_ub=np.zeros((1, 1, 3)), b_eq=1)

    def test_basic_artificial_vars(self):
        # Test if linprog succeeds when at the end of Phase 1 some artificial
        # variables remain basic, and the row in T corresponding to the
        # artificial variables is not all zero.
        c = np.array([-0.1, -0.07, 0.004, 0.004, 0.004, 0.004])
        A_ub = np.array([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0],
                         [0, -1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0],
                         [1.0, 1.0, 0, 0, 0, 0]])
        b_ub = np.array([3.0, 3.0, 3.0, 3.0, 20.0])
        A_eq = np.array([[1.0, 0, -1, 1, -1, 1], [0, -1.0, -1, 1, -1, 1]])
        b_eq = np.array([0, 0])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=0, desired_x=np.zeros_like(c),
                        atol=2e-6)

    def test_empty_constraint_2(self):
        res = linprog([1, -1, 1, -1],
                      bounds=[(0, np.inf), (-np.inf, 0), (-1, 1), (-1, 1)],
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[0, 0, -1, 1], desired_fun=-2)

    def test_zero_row_2(self):
        A_eq = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        b_eq = [0, 3, 0]
        c = [1, 2, 3]
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=3)

    def test_zero_row_4(self):
        A_ub = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        b_ub = [0, 3, 0]
        c = [1, 2, 3]
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=0)

    def test_zero_column_1(self):
        m, n = 3, 4
        np.random.seed(0)
        c = np.random.rand(n)
        c[1] = 1
        A_eq = np.random.rand(m, n)
        A_eq[:, 1] = 0
        b_eq = np.random.rand(m)
        A_ub = [[1, 0, 1, 1]]
        b_ub = 3
        res = linprog(c, A_ub, b_ub, A_eq, b_eq,
                      bounds=[(-10, 10), (-10, 10),
                              (-10, None), (None, None)],
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=-9.7087836730413404)

    def test_singleton_row_eq_2(self):
        c = [1, 1, 1, 2]
        A_eq = [[1, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]]
        b_eq = [1, 2, 1, 4]
        res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=4)

    def test_singleton_row_ub_2(self):
        c = [1, 1, 1, 2]
        A_ub = [[1, 0, 0, 0], [0, 2, 0, 0], [-1, 0, 0, 0], [1, 1, 1, 1]]
        b_ub = [1, 2, -0.5, 4]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (0, None), (0, None), (0, None)],
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=0.5)

    def test_remove_redundancy_infeasibility(self):
        m, n = 10, 10
        c = np.random.rand(n)
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A0[-1, :] = 2 * A0[-2, :]
        b0[-1] *= -1
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            res = linprog(c, A_eq=A0, b_eq=b0,
                          method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_bounded_below_only(self):
        A = np.eye(3)
        b = np.array([1, 2, 3])
        c = np.ones(3)
        res = linprog(c, A_eq=A, b_eq=b, bounds=(0.5, np.inf),
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=b, desired_fun=np.sum(b))

    def test_bounded_above_only(self):
        A = np.eye(3)
        b = np.array([1, 2, 3])
        c = np.ones(3)
        res = linprog(c, A_eq=A, b_eq=b, bounds=(-np.inf, 4),
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=b, desired_fun=np.sum(b))

    def test_unbounded_below_and_above(self):
        A = np.eye(3)
        b = np.array([1, 2, 3])
        c = np.ones(3)
        res = linprog(c, A_eq=A, b_eq=b, bounds=(-np.inf, np.inf),
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=b, desired_fun=np.sum(b))

    def test_bounds_equal_but_infeasible(self):
        c = [-4, 1]
        A_ub = [[7, -2], [0, 1], [2, -2]]
        b_ub = [14, 0, 3]
        bounds = [(2, 2), (0, None)]
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_bounds_equal_but_infeasible2(self):
        c = [-4, 1]
        A_eq = [[7, -2], [0, 1], [2, -2]]
        b_eq = [14, 0, 3]
        bounds = [(2, 2), (0, None)]
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_empty_constraint_1(self):
        res = linprog([-1, 1, -1, 1],
                      bounds=[(0, np.inf), (-np.inf, 0), (-1, 1), (-1, 1)],
                      method=self.method, options=self.options)
        _assert_unbounded(res)
        # Infeasibility detected in presolve requiring no iterations
        # if presolve is not used nit > 0 is expected.
        n = 0 if self.options.get('presolve', True) else 2
        assert_equal(res.nit, n)

    def test_singleton_row_eq_1(self):
        c = [1, 1, 1, 2]
        A_eq = [[1, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]]
        b_eq = [1, 2, 2, 4]
        res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)
        _assert_infeasible(res)
        # Infeasibility detected in presolve requiring no iterations
        # if presolve is not used nit > 0 is expected.
        n = 0 if self.options.get('presolve', True) else 3
        assert_equal(res.nit, n)

    def test_singleton_row_ub_1(self):
        c = [1, 1, 1, 2]
        A_ub = [[1, 0, 0, 0], [0, 2, 0, 0], [-1, 0, 0, 0], [1, 1, 1, 1]]
        b_ub = [1, 2, -2, 4]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (0, None), (0, None), (0, None)],
                      method=self.method, options=self.options)
        _assert_infeasible(res)

        # Infeasibility detected in presolve requiring no iterations
        # if presolve is not used nit > 0 is expected.
        n = 0 if self.options.get('presolve', True) else 3
        assert_equal(res.nit, n)

    def test_zero_column_2(self):
        np.random.seed(0)
        m, n = 2, 4
        c = np.random.rand(n)
        c[1] = -1
        A_eq = np.random.rand(m, n)
        A_eq[:, 1] = 0
        b_eq = np.random.rand(m)

        A_ub = np.random.rand(m, n)
        A_ub[:, 1] = 0
        b_ub = np.random.rand(m)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(None, None),
                      method=self.method, options=self.options)
        _assert_unbounded(res)
        # Infeasibility detected in presolve requiring no iterations
        # if presolve is not used nit > 0 is expected.
        n = 0 if self.options.get('presolve', True) else 5
        assert_equal(res.nit, n)

    def test_zero_row_1(self):
        m, n = 2, 4
        c = np.random.rand(n)
        A_eq = np.random.rand(m, n)
        A_eq[0, :] = 0
        b_eq = np.random.rand(m)
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

        # Infeasibility detected in presolve requiring no iterations
        # if presolve is not used nit > 0 is expected.
        n = 0 if self.options.get('presolve', True) else 1
        assert_equal(res.nit, n)

    def test_zero_row_3(self):
        # detected in presolve?
        m, n = 2, 4
        c = np.random.rand(n)
        A_ub = np.random.rand(m, n)
        A_ub[0, :] = 0
        b_ub = -np.random.rand(m)
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub,
                      method=self.method, options=self.options)
        _assert_infeasible(res)
        assert_equal(res.nit, 0)

    def test_infeasible_ub(self):
        c = [1]
        A_ub = [[2]]
        b_ub = 4
        bounds = (5, 6)
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method=self.method, options=self.options)

        _assert_infeasible(res)
        # Infeasibility detected in presolve requiring no iterations
        # if presolve is not used nit > 0 is expected.
        n = 0 if self.options.get('presolve', True) else 1
        assert_equal(res.nit, n)

    def test_type_error(self):
        c = [1]
        A_eq = [[1]]
        b_eq = "hello"
        assert_raises(TypeError, linprog,
                      c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)

    def test_equal_bounds_no_presolve(self):
        # There was a bug when a lower and upper bound were equal but
        # presolve was not on to eliminate the variable. The bound
        # was being converted to an equality constraint, but the bound
        # was not eliminated, leading to issues in postprocessing.
        c = [1, 2]
        A_ub = [[1, 2], [1.1, 2.2]]
        b_ub = [4, 8]
        bounds = [(1, 2), (2, 2)]
        o = {key: self.options[key] for key in self.options}
        o["presolve"] = False
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method=self.method, options=o)
        _assert_infeasible(res)

    def test_unbounded_below_no_presolve_corrected(self):
        c = [1]
        bounds = [(None, 1)]
        o = {key: self.options[key] for key in self.options}
        o["presolve"] = False
        res = linprog(c=c, bounds=bounds,
                      method=self.method,
                      options=o)
        _assert_unbounded(res)

    def test_unbounded_no_nontrivial_constraints_1(self):
        """
        Test whether presolve pathway for detecting unboundedness after
        constraint elimination is working.
        """
        c = np.array([0, 0, 0, 1, -1, -1])
        A = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1]])
        b = np.array([2, -2, 0])
        bounds = [(None, None), (None, None), (None, None),
                  (-1, 1), (-1, 1), (0, None)]
        res = linprog(c, A, b, None, None, bounds, method=self.method,
                      options=self.options)
        _assert_unbounded(res)
        assert_equal(res.x[-1], np.inf)
        assert_equal(res.message[:36], "The problem is (trivially) unbounded")

    def test_unbounded_no_nontrivial_constraints_2(self):
        """
        Test whether presolve pathway for detecting unboundedness after
        constraint elimination is working.
        """
        c = np.array([0, 0, 0, 1, -1, 1])
        A = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1]])
        b = np.array([2, -2, 0])
        bounds = [(None, None), (None, None), (None, None),
                  (-1, 1), (-1, 1), (None, 0)]
        res = linprog(c, A, b, None, None, bounds, method=self.method,
                      options=self.options)
        _assert_unbounded(res)
        assert_equal(res.x[-1], -np.inf)
        assert_equal(res.message[:36], "The problem is (trivially) unbounded")

    def test_bug_5400(self):
        # https://github.com/scipy/scipy/issues/5400
        bounds = [
            (0, None),
            (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100),
            (0, 900), (0, 900), (0, 900), (0, 900), (0, 900), (0, 900),
            (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]

        f = 1 / 9
        g = -1e4
        h = -3.1
        A_ub = np.array([
            [1, -2.99, 0, 0, -3, 0, 0, 0, -1, -1, 0, -1, -1, 1, 1, 0, 0, 0, 0],
            [1, 0, -2.9, h, 0, -3, 0, -1, 0, 0, -1, 0, -1, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, h, 0, 0, -3, -1, -1, 0, -1, -1, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 1.99, -1, -1, 0, 0, 0, -1, f, f, 0, 0, 0, g, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, -1, -1, 0, 0, 0, -1, f, f, 0, g, 0, 0, 0, 0],
            [0, -1, 1.9, 2.1, 0, 0, 0, f, -1, -1, 0, 0, 0, 0, 0, g, 0, 0, 0],
            [0, 0, 0, 0, -1, 2, -1, 0, 0, 0, f, -1, f, 0, 0, 0, g, 0, 0],
            [0, -1, -1, 2.1, 0, 0, 0, f, f, -1, 0, 0, 0, 0, 0, 0, 0, g, 0],
            [0, 0, 0, 0, -1, -1, 2, 0, 0, 0, f, f, -1, 0, 0, 0, 0, 0, g]])

        b_ub = np.array([
            0.0, 0, 0, 100, 100, 100, 100, 100, 100, 900, 900, 900, 900, 900,
            900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        c = np.array([-1.0, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

        if self.method == 'simplex':
            with pytest.warns(OptimizeWarning):
                res = linprog(c, A_ub, b_ub, bounds=bounds,
                    method=self.method, options=self.options)
        else:
            res = linprog(c, A_ub, b_ub, bounds=bounds,
                    method=self.method, options=self.options)
            _assert_success(res, desired_fun=-106.63507541835018)

    def test_issue_6139(self):
        # Linprog(method='simplex') fails to find a basic feasible solution
        # if phase 1 pseudo-objective function is outside the provided tol.
        # https://github.com/scipy/scipy/issues/6139

        # Note: This is not strictly a bug as the default tolerance determines
        # if a result is "close enough" to zero and should not be expected
        # to work for all cases.

        c = np.array([1, 1, 1])
        A_eq = np.array([[1., 0., 0.], [-1000., 0., - 1000.]])
        b_eq = np.array([5.00000000e+00, -1.00000000e+04])
        A_ub = -np.array([[0., 1000000., 1010000.]])
        b_ub = -np.array([10000000.])
        bounds = (None, None)

        res = linprog(
            c, A_ub, b_ub, A_eq, b_eq, method=self.method,
            bounds=bounds, options=self.options
            )

        _assert_success(
            res, desired_fun=14.95, desired_x=np.array([5, 4.95, 5])
        )

    def test_bug_6690(self):
        # SciPy violates bound constraint despite result status being success
        # when the simplex method is used.
        # https://github.com/scipy/scipy/issues/6690

        A_eq = np.array([[0, 0, 0, 0.93, 0, 0.65, 0, 0, 0.83, 0]])
        b_eq = np.array([0.9626])
        A_ub = np.array([
            [0, 0, 0, 1.18, 0, 0, 0, -0.2, 0, -0.22],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.43, 0, 0, 0, 0, 0, 0],
            [0, -1.22, -0.25, 0, 0, 0, -2.06, 0, 0, 1.37],
            [0, 0, 0, 0, 0, 0, 0, -0.25, 0, 0]
        ])
        b_ub = np.array([0.615, 0, 0.172, -0.869, -0.022])
        bounds = np.array([
            [-0.84, -0.97, 0.34, 0.4, -0.33, -0.74, 0.47, 0.09, -1.45, -0.73],
            [0.37, 0.02, 2.86, 0.86, 1.18, 0.5, 1.76, 0.17, 0.32, -0.15]
        ]).T
        c = np.array([
            -1.64, 0.7, 1.8, -1.06, -1.16, 0.26, 2.13, 1.53, 0.66, 0.28
            ])

        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "Solving system with option 'sym_pos'")
            res = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method=self.method, options=self.options
            )

        desired_fun = -1.19099999999
        desired_x = np.array([
            0.3700, -0.9700, 0.3400, 0.4000, 1.1800,
            0.5000, 0.4700, 0.0900, 0.3200, -0.7300
        ])
        _assert_success(
            res,
            desired_fun=desired_fun,
            desired_x=desired_x
        )

        # Add small tol value to ensure arrays are less than or equal.
        atol = 1e-6
        assert_array_less(bounds[:, 0] - atol, res.x)
        assert_array_less(res.x, bounds[:, 1] + atol)

    def test_bug_7044(self):
        # linprog fails to identify correct constraints with simplex method
        # leading to a non-optimal solution if A is rank-deficient.
        # https://github.com/scipy/scipy/issues/7044

        A, b, c, N = magic_square(3)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            res = linprog(c, A_eq=A, b_eq=b,
                          method=self.method, options=self.options)

        desired_fun = 1.730550597
        _assert_success(res, desired_fun=desired_fun)
        assert_allclose(A.dot(res.x), b)
        assert_array_less(np.zeros(res.x.size) - 1e-5, res.x)

    def test_issue_7237(self):
        # https://github.com/scipy/scipy/issues/7237
        # The simplex method sometimes "explodes" if the pivot value is very
        # close to zero.

        c = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0])
        A_ub = np.array([
            [1., -724., 911., -551., -555., -896., 478., -80., -293.],
            [1., 566., 42., 937.,233., 883., 392., -909., 57.],
            [1., -208., -894., 539., 321., 532., -924., 942., 55.],
            [1., 857., -859., 83., 462., -265., -971., 826., 482.],
            [1., 314., -424., 245., -424., 194., -443., -104., -429.],
            [1., 540., 679., 361., 149., -827., 876., 633., 302.],
            [0., -1., -0., -0., -0., -0., -0., -0., -0.],
            [0., -0., -1., -0., -0., -0., -0., -0., -0.],
            [0., -0., -0., -1., -0., -0., -0., -0., -0.],
            [0., -0., -0., -0., -1., -0., -0., -0., -0.],
            [0., -0., -0., -0., -0., -1., -0., -0., -0.],
            [0., -0., -0., -0., -0., -0., -1., -0., -0.],
            [0., -0., -0., -0., -0., -0., -0., -1., -0.],
            [0., -0., -0., -0., -0., -0., -0., -0., -1.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1.]
            ])
        b_ub = np.array([
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.])
        A_eq = np.array([[0., 1., 1., 1., 1., 1., 1., 1., 1.]])
        b_eq = np.array([[1.]])
        bounds = [(None, None)] * 9

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method=self.method, options=self.options)
        _assert_success(res, desired_fun=108.568535, atol=1e-6)

    def test_issue_8174(self):
        # https://github.com/scipy/scipy/issues/8174
        # The simplex method sometimes "explodes" if the pivot value is very
        # close to zero.
        A_ub = np.array([
            [22714, 1008, 13380, -2713.5, -1116],
            [-4986, -1092, -31220, 17386.5, 684],
            [-4986, 0, 0, -2713.5, 0],
            [22714, 0, 0, 17386.5, 0]])
        b_ub = np.zeros(A_ub.shape[0])
        c = -np.ones(A_ub.shape[1])
        bounds = [(0,1)] * A_ub.shape[1]

        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
            options=self.options, method=self.method)

    def test_issue_8174_stackoverflow(self):
        # Test supplementary example from issue 8174.
        # https://github.com/scipy/scipy/issues/8174
        # https://stackoverflow.com/questions/47717012/linprog-in-scipy-optimize-checking-solution
        c = np.array([1, 0, 0, 0, 0, 0, 0])
        A_ub = -np.identity(7)
        b_ub = np.array([[-2],[-2],[-2],[-2],[-2],[-2],[-2]])
        A_eq = np.array([
            [1, 1, 1, 1, 1, 1, 0],
            [0.3, 1.3, 0.9, 0, 0, 0, -1],
            [0.3, 0, 0, 0, 0, 0, -2/3],
            [0, 0.65, 0, 0, 0, 0, -1/15],
            [0, 0, 0.3, 0, 0, 0, -1/15]
        ])
        b_eq = np.array([[100],[0],[0],[0],[0]])

        with pytest.warns(OptimizeWarning):
            res = linprog(
                c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                method=self.method, options=self.options
            )
        _assert_success(res, desired_fun=43.3333333331385)

    def test_bug_8662(self):
        # scipy.linprog returns incorrect optimal result for constraints using
        # default bounds, but, correct if boundary condition as constraint.
        # https://github.com/scipy/scipy/issues/8662
        c = [-10, 10, 6, 3]
        A = [
            [8, -8, -4, 6],
            [-8, 8, 4, -6],
            [-4, 4, 8, -4],
            [3, -3, -3, -10]
        ]
        b = [9, -9, -9, -4]
        bounds = [(0, None), (0, None), (0, None), (0, None)]
        desired_fun = 36.0000000000

        res1 = linprog(c, A, b, bounds=bounds,
                       method=self.method, options=self.options)

        # Set boundary condition as a constraint
        A.append([0, 0, -1, 0])
        b.append(0)
        bounds[2] = (None, None)

        res2 = linprog(c, A, b, bounds=bounds, method=self.method,
                       options=self.options)
        rtol = 1e-5
        _assert_success(res1, desired_fun=desired_fun, rtol=rtol)
        _assert_success(res2, desired_fun=desired_fun, rtol=rtol)

    def test_bug_8663(self):
        A = [[0, -7]]
        b = [-6]
        c = [1, 5]
        bounds = [(0, None), (None, None)]
        res = linprog(c, A_eq=A, b_eq=b, bounds=bounds,
                      method=self.method, options=self.options)
        _assert_success(res,
                        desired_x=[0, 6./7],
                        desired_fun=5*6./7)

    def test_bug_8664(self):
        # Weak test. Ideally should _detect infeasibility_ for all options.
        c = [4]
        A_ub = [[2], [5]]
        b_ub = [4, 4]
        A_eq = [[0], [-8], [9]]
        b_eq = [3, 2, 10]
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sup.filter(OptimizeWarning, "Solving system with option...")
            o = {key: self.options[key] for key in self.options}
            o["presolve"] = False
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, options=o,
                          method=self.method)
        assert_(not res.success, "incorrectly reported success")

    def test_bug_8973(self):
        """
        Test whether bug described at:
        https://github.com/scipy/scipy/issues/8973
        was fixed.
        """
        c = np.array([0, 0, 0, 1, -1])
        A = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        b = np.array([2, -2])
        bounds = [(None, None), (None, None), (None, None), (-1, 1), (-1, 1)]
        res = linprog(c, A, b, None, None, bounds, method=self.method,
                      options=self.options)
        _assert_success(res,
                        desired_x=[2, -2, 0, -1, 1],
                        desired_fun=-2)

    def test_bug_8973_2(self):
        """
        Additional test for:
        https://github.com/scipy/scipy/issues/8973
        suggested in
        https://github.com/scipy/scipy/pull/8985
        review by @antonior92
        """
        c = np.zeros(1)
        A = np.array([[1]])
        b = np.array([-2])
        bounds = (None, None)
        res = linprog(c, A, b, None, None, bounds, method=self.method,
                      options=self.options)
        _assert_success(res)  # would not pass if solution is infeasible


class BaseTestLinprogSimplex(LinprogCommonTests):
    method = "simplex"


class TestLinprogSimplexCommon(BaseTestLinprogSimplex):
    options = {}

    def test_callback(self):
        # Check that callback is as advertised
        last_cb = {}

        def cb(res):
            message = res.pop('message')
            complete = res.pop('complete')

            assert_(res.pop('phase') in (1, 2))
            assert_(res.pop('status') in range(4))
            assert_(isinstance(res.pop('nit'), int))
            assert_(isinstance(complete, bool))
            assert_(isinstance(message, str))

            if complete:
                last_cb['x'] = res['x']
                last_cb['fun'] = res['fun']
                last_cb['slack'] = res['slack']
                last_cb['con'] = res['con']

        c = np.array([-3, -2])
        A_ub = [[2, 1], [1, 1], [1, 0]]
        b_ub = [10, 8, 4]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, callback=cb, method=self.method)

        _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])
        assert_allclose(last_cb['fun'], res['fun'])
        assert_allclose(last_cb['x'], res['x'])
        assert_allclose(last_cb['con'], res['con'])
        assert_allclose(last_cb['slack'], res['slack'])

    def test_issue_7237(self):
        with pytest.raises(ValueError):
            super(TestLinprogSimplexCommon, self).test_issue_7237()

    def test_issue_8174(self):
        with pytest.warns(OptimizeWarning):
            super(TestLinprogSimplexCommon, self).test_issue_8174()


class TestLinprogSimplexBland(BaseTestLinprogSimplex):
    options = {'bland': True}

    def test_bug_5400(self):
        with pytest.raises(ValueError):
            super(TestLinprogSimplexBland, self).test_bug_5400()

    def test_issue_8174(self):
        with pytest.warns(OptimizeWarning):
            super(TestLinprogSimplexBland, self).test_issue_8174()


class TestLinprogSimplexNoPresolve(BaseTestLinprogSimplex):
    options = {'presolve': False}

    def test_issue_6139(self):
        # Linprog(method='simplex') fails to find a basic feasible solution
        # if phase 1 pseudo-objective function is outside the provided tol.
        # https://github.com/scipy/scipy/issues/6139
        # Without ``presolve`` eliminating such rows the result is incorrect.
        with pytest.raises(ValueError):
            return super(TestLinprogSimplexNoPresolve, self).test_issue_6139()

    def test_issue_7237(self):
        with pytest.raises(ValueError):
            super(TestLinprogSimplexNoPresolve, self).test_issue_7237()

    def test_issue_8174(self):
        with pytest.warns(OptimizeWarning):
            super(TestLinprogSimplexNoPresolve, self).test_issue_8174()

    def test_issue_8174_stackoverflow(self):
        # Test expects linprog to raise a warning during presolve.
        # As ``'presolve'=False`` no warning should be raised.
        # Despite not presolving the result is still correct.
        with pytest.warns(OptimizeWarning) as redundant_warning:
            super(TestLinprogSimplexNoPresolve, self).test_issue_8174()

    def test_unbounded_no_nontrivial_constraints_1(self):
        pytest.skip("Tests behavior specific to presolve")

    def test_unbounded_no_nontrivial_constraints_2(self):
        pytest.skip("Tests behavior specific to presolve")


class BaseTestLinprogIP(LinprogCommonTests):
    method = "interior-point"


class TestLinprogIPSpecific(object):
    method = "interior-point"
    # the following tests don't need to be performed separately for
    # sparse presolve, sparse after presolve, and dense

    def test_unbounded_below_no_presolve_original(self):
        # formerly caused segfault in TravisCI w/ "cholesky":True
        c = [-1]
        bounds = [(None, 1)]
        res = linprog(c=c, bounds=bounds,
                      method=self.method,
                      options={"presolve": False, "cholesky": True})
        _assert_success(res, desired_fun=-1)

    def test_cholesky(self):
        # Test with a rather large problem (400 variables,
        # 40 constraints) generated by https://gist.github.com/denis-bz/8647461
        # use cholesky factorization and triangular solves
        A, b, c = lpgen_2d(20, 20)
        res = linprog(c, A_ub=A, b_ub=b, method=self.method,
                      options={"cholesky": True})  # only for dense
        _assert_success(res, desired_fun=-64.049494229)

    def test_alternate_initial_point(self):
        # Test with a rather large problem (400 variables,
        # 40 constraints) generated by https://gist.github.com/denis-bz/8647461
        # use "improved" initial point
        A, b, c = lpgen_2d(20, 20)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "scipy.linalg.solve\nIll...")
            sup.filter(OptimizeWarning, "Solving system with option...")
            res = linprog(c, A_ub=A, b_ub=b, method=self.method,
                          options={"ip": True, "disp": True})
            # ip code is independent of sparse/dense
        _assert_success(res, desired_fun=-64.049494229)

    def test_maxiter(self):
        # Test with a rather large problem (400 variables,
        # 40 constraints) generated by https://gist.github.com/denis-bz/8647461
        # test iteration limit
        A, b, c = lpgen_2d(20, 20)
        maxiter = np.random.randint(6) + 1  # problem takes 7 iterations
        res = linprog(c, A_ub=A, b_ub=b, method=self.method,
                      options={"maxiter": maxiter})
        # maxiter is independent of sparse/dense
        assert_equal(res.status, 1)
        assert_equal(res.nit, maxiter)

    def test_disp(self):
        # Test with a rather large problem (400 variables,
        # 40 constraints) generated by https://gist.github.com/denis-bz/8647461
        # test that display option does not break anything.
        A, b, c = lpgen_2d(20, 20)
        res = linprog(c, A_ub=A, b_ub=b, method=self.method,
                      options={"disp": True})
        # disp is independent of sparse/dense
        _assert_success(res, desired_fun=-64.049494229)

    def test_callback(self):
        def f():
            pass

        A = [[0, -7]]
        b = [-6]
        c = [1, 5]
        bounds = [(0, None), (None, None)]

        # Linprog should solve in presolve. As the interior-point method is
        # not used the the callback should never be needed and no error
        # returned
        res = linprog(c, A_eq=A, b_eq=b, bounds=bounds,
                      method=self.method, callback=f)
        _assert_success(res, desired_x=[0, 6./7], desired_fun=5*6./7)

        # Without presolve the solver reverts to the interior-point method
        # Interior-point currently does not implement callback functions.
        with pytest.raises(NotImplementedError):
            res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method=self.method,
                          callback=f, options={'presolve': False})


class TestLinprogIPSparse(BaseTestLinprogIP):
    options = {"sparse": True}

    @pytest.mark.xfail(reason='Fails with ATLAS, see gh-7877')
    def test_bug_6690(self):
        # Test defined in base class, but can't mark as xfail there
        super(TestLinprogIPSparse, self).test_bug_6690()

    def test_magic_square_sparse_no_presolve(self):
        # test linprog with a problem with a rank-deficient A_eq matrix
        A, b, c, N = magic_square(3)
        with suppress_warnings() as sup:
            sup.filter(MatrixRankWarning, "Matrix is exactly singular")
            sup.filter(OptimizeWarning, "Solving system with option...")
            o = {key: self.options[key] for key in self.options}
            o["presolve"] = False
            res = linprog(c, A_eq=A, b_eq=b, bounds=(0, 1),
                          options=o, method=self.method)
        _assert_success(res, desired_fun=1.730550597)

    def test_sparse_solve_options(self):
        A, b, c, N = magic_square(3)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            sup.filter(OptimizeWarning, "Invalid permc_spec option")
            o = {key: self.options[key] for key in self.options}
            permc_specs = ('NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A',
                           'COLAMD', 'ekki-ekki-ekki')
            for permc_spec in permc_specs:
                o["permc_spec"] = permc_spec
                res = linprog(c, A_eq=A, b_eq=b, bounds=(0, 1),
                              method=self.method, options=o)
                _assert_success(res, desired_fun=1.730550597)


class TestLinprogIPDense(BaseTestLinprogIP):
    options = {"sparse": False}


class TestLinprogIPSparsePresolve(BaseTestLinprogIP):
    options = {"sparse": True, "_sparse_presolve": True}

    def test_enzo_example_c_with_infeasibility(self):
        pytest.skip('_sparse_presolve=True incompatible with presolve=False')

    @pytest.mark.xfail(reason='Fails with ATLAS, see gh-7877')
    def test_bug_6690(self):
        # Test defined in base class, but can't mark as xfail there
        super(TestLinprogIPSparsePresolve, self).test_bug_6690()


def test_unknown_solver():
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]

    assert_raises(ValueError, linprog,
                  c, A_ub=A_ub, b_ub=b_ub, method='ekki-ekki-ekki')
