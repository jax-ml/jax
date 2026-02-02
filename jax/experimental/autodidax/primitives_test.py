"""Tests for autodidax primitives and JVP."""

import math
import unittest
from jax.experimental.autodidax import primitives as P
from jax.experimental.autodidax import core


class PrimitivesEvalTest(unittest.TestCase):
    """Tests for primitive evaluation."""

    def test_add(self):
        self.assertEqual(P.add(1.0, 2.0), 3.0)

    def test_mul(self):
        self.assertEqual(P.mul(2.0, 3.0), 6.0)

    def test_neg(self):
        self.assertEqual(P.neg(5.0), -5.0)

    def test_sin(self):
        self.assertAlmostEqual(P.sin(0.0), 0.0)
        self.assertAlmostEqual(P.sin(math.pi / 2), 1.0, places=10)

    def test_cos(self):
        self.assertAlmostEqual(P.cos(0.0), 1.0)
        self.assertAlmostEqual(P.cos(math.pi), -1.0, places=10)

    def test_composite_function(self):
        def f(x):
            y = P.sin(x)
            z = P.mul(y, 2.0)
            w = P.neg(z)
            return P.add(w, x)

        result = f(3.0)
        expected = 3.0 - 2.0 * math.sin(3.0)
        self.assertAlmostEqual(result, expected, places=10)


class JVPTest(unittest.TestCase):
    """Tests for forward-mode autodiff (JVP)."""

    def test_jvp_add(self):
        def f(x):
            return P.add(x, 1.0)

        primal, tangent = core.jvp(f, (3.0,), (1.0,))
        self.assertEqual(primal, 4.0)
        self.assertEqual(tangent, 1.0)

    def test_jvp_mul(self):
        def f(x):
            return P.mul(x, x)

        primal, tangent = core.jvp(f, (3.0,), (1.0,))
        self.assertEqual(primal, 9.0)
        self.assertEqual(tangent, 6.0)

    def test_jvp_sin(self):
        def f(x):
            return P.sin(x)

        primal, tangent = core.jvp(f, (3.0,), (1.0,))
        self.assertAlmostEqual(primal, math.sin(3.0), places=10)
        self.assertAlmostEqual(tangent, math.cos(3.0), places=10)

    def test_jvp_cos(self):
        def f(x):
            return P.cos(x)

        primal, tangent = core.jvp(f, (0.0,), (1.0,))
        self.assertAlmostEqual(primal, 1.0, places=10)
        self.assertAlmostEqual(tangent, 0.0, places=10)

    def test_jvp_composite(self):
        def f(x):
            y = P.sin(x)
            z = P.mul(y, 2.0)
            return z

        primal, tangent = core.jvp(f, (3.0,), (1.0,))
        self.assertAlmostEqual(primal, 2.0 * math.sin(3.0), places=10)
        self.assertAlmostEqual(tangent, 2.0 * math.cos(3.0), places=10)

    def test_derivative_helper(self):
        def deriv(f):
            return lambda x: core.jvp(f, (x,), (1.0,))[1]

        def f(x):
            return P.sin(x)

        df = deriv(f)
        self.assertAlmostEqual(df(3.0), math.cos(3.0), places=10)

    def test_second_derivative(self):
        """Test nested JVP for computing second derivatives."""
        def deriv(f):
            return lambda x: core.jvp(f, (x,), (1.0,))[1]

        def f(x):
            return P.sin(x)

        ddf = deriv(deriv(f))
        self.assertAlmostEqual(ddf(3.0), -math.sin(3.0), places=10)

    def test_third_derivative(self):
        """Test computing third derivatives via nested JVP."""
        def deriv(f):
            return lambda x: core.jvp(f, (x,), (1.0,))[1]

        def f(x):
            return P.sin(x)

        dddf = deriv(deriv(deriv(f)))
        self.assertAlmostEqual(dddf(3.0), -math.cos(3.0), places=10)


class MakeJaxprTest(unittest.TestCase):
    """Tests for make_jaxpr IR building."""

    def test_make_jaxpr_simple(self):
        """Test tracing a simple function to jaxpr."""
        def f(x):
            return P.mul(x, x)

        jaxpr = core.make_jaxpr(f)(3.0)
        jaxpr_str = str(jaxpr)
        print(f"Jaxpr for f(x) = x*x:\n{jaxpr_str}")
        self.assertIn("mul", jaxpr_str)
        self.assertIn("lambda", jaxpr_str)

    def test_make_jaxpr_composite(self):
        """Test tracing a composite function to jaxpr."""
        def f(x):
            y = P.sin(x)
            z = P.mul(y, 2.0)
            return z

        jaxpr = core.make_jaxpr(f)(3.0)
        jaxpr_str = str(jaxpr)
        print(f"\nJaxpr for f(x) = sin(x) * 2:\n{jaxpr_str}")
        self.assertIn("sin", jaxpr_str)
        self.assertIn("mul", jaxpr_str)

    def test_make_jaxpr_two_args(self):
        """Test tracing a two-argument function."""
        def f(x, y):
            return P.add(P.mul(x, y), x)

        jaxpr = core.make_jaxpr(f)(3.0, 4.0)
        jaxpr_str = str(jaxpr)
        print(f"\nJaxpr for f(x, y) = x*y + x:\n{jaxpr_str}")
        self.assertIn("mul", jaxpr_str)
        self.assertIn("add", jaxpr_str)


if __name__ == "__main__":
    unittest.main()
