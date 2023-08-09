import unittest 
import jax._src.test_util as jtu
import jax.numpy as jnp
from jax.scipy.special import bessel_j0, bessel_j1, bessel_jn

from scipy.special import jv

class TestBesselFunctions(unittest.TestCase):
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

    def test_bessel_jn(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        v = 2
        expected_output = jv(v, x)
        output = bessel_jn(x,v)
        self.assertTrue(jnp.allclose(output, expected_output))

if __name__ == '__main__':
    unittest.main()