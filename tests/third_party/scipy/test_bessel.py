import unittest
import jax._src.test_util as jtu
import jax.numpy as jnp
from jax.scipy.special import bessel_j0, bessel_j1, bessel_jv

from scipy.special import jv

class TestBesselFunctions(jtu.JaxTestCase):
    def test_bessel_j0(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        expected_output = jv(0, x)
        output = bessel_j0(x)
        self.assertTrue(jnp.allclose(output, expected_output))

    def test_bessel_j1(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        expected_output = jv(1, x)
        output = bessel_j1(x)
        self.assertTrue(jnp.allclose(output, expected_output))

    def test_bessel_jv(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        v = 2
        expected_output = jv(2, x)
        output = bessel_jv(v, x)
        self.assertTrue(jnp.allclose(output, expected_output))

if __name__ == '__main__':
    unittest.main()