import jax.numpy as np
from jax import jvp, jet, grad


def f(z, t):
  return 4 * z / t


# Initial Conditions
t0, z0 = 2., 0.5


# True solution
def sol(t):
  return (z0 / t0**4) * t**4


# Derivatives at t_eval
t_eval = 4.
z_eval = sol(t_eval)

# True total derivatives
dz = grad(sol)(t_eval)
ddz = grad(grad(sol))(t_eval)
dddz = grad(grad(grad(sol)))(t_eval)

# Jet total derivatives
z_terms = (np.ones_like(z_eval),np.zeros_like(z_eval))
t_terms = (np.ones_like(t_eval),np.zeros_like(t_eval))

jet(f,(z_eval,t_eval),np.array([z_terms,t_terms]))

z_eval
