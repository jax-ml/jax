import jax.numpy as np
from jax import jvp, jet, grad
from scipy.integrate import odeint


def shift_jet(new_primal, old_jet):
  return (new_primal, [old_jet[0]] + old_jet[1])


def std_terms(x, n):
  return [np.ones_like(x)] + [np.zeros_like(x)] * (n - 1)


def zero_terms(x, n):
  return [np.zeros_like(x)] * n


# XXX Fails without pow and div
def test_sol_divpow():
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
  order = 4
  z_terms = [np.ones_like(z_eval)] + [np.zeros_like(z_eval)] * (order - 1)
  t_terms = [np.ones_like(t_eval)] + [np.zeros_like(t_eval)] * (order - 1)

  jet(f, (z_eval, t_eval), [z_terms, t_terms])


def test_sol_sin_exp():
  def f(z, t):
    return z * np.sin(t)

  # Initial Conditions
  t0, z0 = 0., 1.

  # True solution
  def sol(t):
    return np.exp(-np.cos(t) + np.cos(t0)) * z0

  # Want derivatives at t_eval
  t_eval = 4.
  z_eval = sol(t_eval)

  # Confirm the sol is true solution
  assert np.isclose(odeint(f, z0, [t0, t_eval])[1], z_eval)

  # AD total through sol
  dz = grad(sol)(t_eval)
  ddz = grad(grad(sol))(t_eval)
  dddz = grad(grad(grad(sol)))(t_eval)

  # Jet total derivatives
  order = 4
  z_terms = std_terms(z_eval,order)
  t_terms = std_terms(t_eval,order)
  j1 = jet(f, (z_eval, t_eval), [z_terms, t_terms])

  zhat, zhat_terms = shift_jet(z_eval, j1)
  j2 = jet(f, (zhat, t_eval), (zhat_terms, t_terms2))
  j2

  # Jet 1: z,t terms standard
  order = 4
  z_terms = std_terms(z_eval,order)
  t_terms = std_terms(t_eval,order)
  j1 = jet(f, (z_eval, t_eval), [z_terms, t_terms])
  j1

  # Jet 2: z terms standard, t terms zero
  order = 4
  z_terms = std_terms(z_eval,order)
  t_terms = zero_terms(t_eval,order)
  j2 = jet(f, (z_eval, t_eval), [z_terms, t_terms])
  j2

  # Jet 3: z terms zero, t terms standard
  order = 4
  z_terms = zero_terms(z_eval,order)
  t_terms = std_terms(t_eval,order)
  j3 = jet(f, (z_eval, t_eval), [z_terms, t_terms])
  j3

  # Jet 4: z terms shifted j1, t terms standard
  order = 4
  z_hat, z_terms = shift_jet(z_eval,j1)
  t_terms = std_terms(t_eval,order+1)
  j4 = jet(f, (z_hat, t_eval), [z_terms, t_terms])
  j4

  # Jet 5: z terms shifted j1, t terms zero
  order = 4
  z_hat, z_terms = shift_jet(z_eval,j3)
  t_terms = std_terms(t_eval,order+1)
  j4 = jet(f, (z_hat, t_eval), [z_terms, t_terms])
  j4

def test_sol_auton():

  def f(z,t):
    return 0.5*z
  def g(z):
    return f(z,0.) #since autonomous

  # Initial Conditions
  t0, z0 = 0., 1.

  # True solution
  def sol(t):
    return np.exp(0.5*t - 0.5*t0) * z0

  # Want derivatives at t_eval
  t_eval = 2.
  z_eval = sol(t_eval)

  # Confirm the sol is true solution
  assert np.isclose(odeint(f, z0, [t0, t_eval])[1], z_eval)

  # AD total through sol
  dz = grad(sol)(t_eval)
  ddz = grad(grad(sol))(t_eval)
  dddz = grad(grad(grad(sol)))(t_eval)

  # Jet total derivatives
  order = 4
  z_terms = std_terms(z_eval,order)
  t_terms = std_terms(t_eval,order)
  j1 = jet(f, (z_eval, t_eval), [z_terms, t_terms])

  zhat, zhat_terms = shift_jet(z_eval, j1)
  j2 = jet(f, (zhat, t_eval), (zhat_terms, t_terms2))
  j2

  # Jet 1: z,t terms standard
  order = 4
  z_terms = std_terms(z_eval,order)
  t_terms = std_terms(t_eval,order)
  j1 = jet(f, (z_eval, t_eval), [z_terms, t_terms])

  # Jet 5: z terms shifted j1, t terms zero
  order = 4
  z_terms = std_terms(z_eval,1)
  j1 = jet(g, (z_eval, ), [z_terms, ])
  j_prev = j1
  for i in range(order):
    z_eval, z_terms = shift_jet(z_eval,j_prev)
    print "zt", z_terms
    j_prev = jet(g, (z_hat,), [z_terms, ])
  (j_prev[0],j_prev[1][:order])
