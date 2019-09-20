import numpy.random as npr
from scipy.special import factorial as fact

import jax
from jax.util import safe_map
from jax import vjp, jvp, jet, grad, linearize
import jax.numpy as np

map = safe_map

from jet_test import repeated, jvp_taylor, jvp_test_jet


def sys_1D():
  from scipy.integrate import odeint

  ## IVP with known solution
  def f(z, t):
    return g(z)

  def g(z):
    return z * z * 0.1

  # Initial  Conditions
  t0, z0 = 2., 1.

  # Closed-form solution
  def true_sol(t):
    return -(10. * z0) / (-10. + t * z0 - t0 * z0)

  # Evaluate at t_eval
  t_eval = 5.
  z_eval_true = true_sol(t_eval)

  # Use numerical integrator to test true solution
  z_eval_odeint = odeint(f, z0, [t0, t_eval])[1]
  assert np.isclose(z_eval_true, z_eval_odeint)

  # True derivatives of solution
  true_ds = jvp_taylor(true_sol, (t_eval, ), ((1., 0., 0., 0., 0., 0., 0.), ))

  return g, z_eval_true, true_ds[1]


def sys_2D():
  from scipy.integrate import odeint

  ## IVP with known solution
  def f(z, t):
    return g(z)

  def g(z):
    z0, z1, t = z
    return np.array([np.sin(2. * t), np.cos(2. * t), 1.])

  # Initial  Conditions
  t0, z0 = 2., (0., 0.)

  # Closed-form solution
  def true_sol(t):
    return np.array([
        z0[0] + 0.5 * np.cos(2. * t0) -
        0.5 * np.cos(2. * t) * np.cos(2. * t0) +
        0.5 * np.sin(2. * t) * np.sin(2. * t0), 0.5 *
        (2. * z0[1] + np.sqrt(1. - np.cos(2. * t0)**2) +
         np.cos(2. * t0) * np.sin(2. * t) + np.cos(2. * t) * np.sin(2. * t0)),
        t0 + t
    ])

  # Evaluate at t_eval
  t_eval = 3.
  z_eval_true = true_sol(t_eval)

  # Use numerical integrator to test true solution
  z_eval_odeint = odeint(f, (z0[0], z0[1], t0), [t0, t_eval + t0])[1]
  assert np.allclose(np.array(z_eval_true), z_eval_odeint)

  true_ds = jvp_taylor(true_sol, (t_eval, ), ((1., 0., 0., 0., 0., 0., 0.), ))

  return g, z_eval_true, true_ds[1]


def sys_2Db():
  from scipy.integrate import odeint

  ## IVP with known solution
  def f(z, t):
    return g(z)

  def g(z):
    z0, z1, t = z
    return np.array([np.sin(t) * t, np.cos(2. * t), 1.])

  # Initial  Conditions
  t0, z0 = 2., (0., 0.)

  # Closed-form solution
  def true_sol(t):
    return np.array([
      -1.*( -1. * z0[0] - 1.* t0 * np.cos(t0) + t * np.cos(t)*np.cos(t0) + t0*np.cos(t)*np.cos(t0) + np.sqrt(1-np.cos(t0)**2) - np.cos(t0)*np.sin(t) - np.cos(t)*np.sin(t0) - t * np.sin(t) * np.sin(t0) - t0*np.sin(t)*np.sin(t0)),
        z0[1] + 0.5 * np.cos(2*t0) * np.sin(2*t) - 0.5 * np.sin(2*t0) + 0.5 * np.cos(2*t)*np.sin(2*t0),
        t0 + t
    ])

  # Evaluate at t_eval
  t_eval = 3.
  z_eval_true = true_sol(t_eval)

  # Use numerical integrator to test true solution
  z_eval_odeint = odeint(f, (z0[0], z0[1], t0), [t0, t_eval + t0])[1]
  assert np.allclose(np.array(z_eval_true), z_eval_odeint)

  true_ds = jvp_taylor(true_sol, (t_eval, ), ((1., 0., 0., 0., 0., 0., 0.), ))

  return g, z_eval_true, true_ds[1]

def sys_2Dc():
  from scipy.integrate import odeint

  ## IVP with known solution
  def f(z, t):
    return g(z)

  def g(z):
    z0, z1, t = z
    return np.array([np.sin(t) * z1, np.cos(t), 1.])

  # Initial  Conditions
  t0, z0 = 2., (0., 0.)

  # Closed-form solution
  def true_sol(t):
    return np.array([
      0.5*(1.* t + 2.*z0[0] + 2.*z0[1]*np.cos(t0) - 2.*z0[1]*np.cos(1.*t)*np.cos(t0) - 
       1.*np.cos(t0)*np.sqrt(1. - 1.*np.cos(t0)**2) + 
       2.*np.cos(1.*t)*np.cos(t0)*np.sqrt(1. - 1.*np.cos(t0)**2) - 
       2.*np.cos(1.*t)*np.cos(2*t0)*np.sin(1.*t) + 0.5*np.cos(2* t0)*np.sin(2.*t) - 
       2.*np.cos(1.*t)**2*np.cos(t0)*np.sin(t0) + 2.*z0[1]*np.sin(1.*t)*np.sin(t0) - 
       2.*np.sqrt(1. - 1.*np.cos(t0)**2)*np.sin(1.*t)*np.sin(t0) + 
       0.5*np.cos(2.*t)*np.sin(2*t0) +np.sin(t)**2*np.sin(2*t0)), 
    -1.* (-1.*z0[1] +np.sqrt(1. -np.cos(t0)**2) - 
        1.*np.cos(t0)*np.sin(t) - np.cos(t)*np.sin(t0)),
        t0 + t
    ])

  # Evaluate at t_eval
  t_eval = 3.
  z_eval_true = true_sol(t_eval)

  # Use numerical integrator to test true solution
  z_eval_odeint = odeint(f, (z0[0], z0[1], t0), [t0, t_eval + t0])[1]
  assert np.allclose(np.array(z_eval_true), z_eval_odeint)

  true_ds = jvp_taylor(true_sol, (t_eval, ), ((1., 0., 0., 0., 0., 0., 0.), ))

  return g, z_eval_true, true_ds[1]

def jet_sol_recursive(g, x0):
  (y0, [y1h]) = jet(g, (x0, ), ((np.ones_like(x0), ), ))
  (y0, [y1, y2h]) = jet(g, (x0, ), ((
      y0,
      y1h,
  ), ))
  (y0, [y1, y2, y3h]) = jet(g, (x0, ), ((y0, y1, y2h), ))
  (y0, [y1, y2, y3, y4h]) = jet(g, (x0, ), ((y0, y1, y2, y3h), ))
  (y0, [y1, y2, y3, y4, y5h]) = jet(g, (x0, ), ((y0, y1, y2, y3, y4h), ))
  (y0, [y1, y2, y3, y4, y5, y6h]) = jet(g, (x0, ),
                                        ((y0, y1, y2, y3, y4, y5h), ))
  return [y0, y1, y2, y3, y4]


def jet_sol_multi_jvp(g, x0):
  # first pass
  s = 0
  j = 2**(s + 1) - 1
  (y0h, [y1h]) = jet(g, (x0, ), ((np.zeros_like(x0), ), ))
  # A0 = lambda v: jvp(y0_coeff,(x0,),(v,))[1]
  A0 = lambda v: jvp(lambda x0: jet(g, (x0, ), ((np.zeros_like(x0), ), )), (x0, ), ((v, )))[
      1][0]

  x1 = 1. / 1 * (y0h)
  x2 = 1. / 2 * (y1h + A0(x1))

  #second pass
  s = 1
  j = 2**(s + 1) - 1
  (y0h, [y1h, y2h, y3h, y4h, y5h,
         y6h]) = jet(g, (x0, ), ([x1, x2 * fact(2)]+[np.zeros_like(x1)]*4, ))
  #dy0/dx0*v
  A0 = lambda v: jvp(lambda x0: jet(g, (x0, ), ((np.zeros_like(x1), ), )), (x0, ), ((v, )))[
      1][0]
  #dy1/dx0*v
  A1 = lambda v: jvp(lambda x0: jet(g, (x0, ), ((x1, ), )), (x0, ), ((v, )))[
      1][1][0]
  #dy2/dx0*v
  A2 = lambda v: jvp(lambda x0: jet(g, (x0, ), ((x1, x2), )), (x0, ), ((v, )))[
      1][1][1]

  x3 = 1. / 3 * (y2h / fact(2))
  x4 = 1. / 4 * (y3h / fact(3) + A0(x3))
  x5 = 1. / 5 * (y4h / fact(4) + A1(x3) + A0(x4))
  # x6 = 1. / 6 * (y5h / fact(5) + A2(x3) + A1(x4) + A0(x5))

  return [x1, x2*fact(2), x3*fact(3), x4*fact(4), x5*fact(5)]

def jet_sol_jvp_vjp(g,x0):
  f_jet = lambda x0: jet(g, (x0,), ((np.zeros_like(x0),),))
  yhs, f_vjp = vjp(f_jet, *(x0,))
  y_I = (np.ones_like(yhs[0]), [np.ones_like(yhs[1][0])] * len(yhs[1]))
  y_10 = (np.ones_like(yhs[0]), [np.zeros_like(yhs[1][0])] * len(yhs[1]))
  y_01 = (np.zeros_like(yhs[0]), [np.ones_like(yhs[1][0])] * len(yhs[1]))

  x1 = yhs[0]

  A0_x1 = jvp(f_jet,(x0,),(x1,))[1]
  jet(lambda a,b : a*b, (A0_x1[0],np.ones(3)),( A0_x1[1], [ np.ones(3)]))[0]

  oA0 = f_vjp(y_I)
  sumx2 = oA0[0] * x1
  x2 = yhs[1][0] + sumx2

  import ipdb; ipdb.set_trace()


  return [x1, x2 * fact(2)]

  # fwd = jvp(f,primals,tangents_in)[1] * tangents_out
  # rev = np.sum(np.array(vjp(f,*primals)[1](tangents_out)) * np.array([tangents_in]))

def newton2_worked_w_scalar(g,x0):
  zero_term = np.zeros_like(x0)
  f_jet = lambda x0: jet(g, (x0,), ((zero_term,),))
  yhs, f_vjp = vjp(f_jet, *(x0,))


  x1 = 1./1 * yhs[0]
  x2 = 1./2 * (yhs[1][0] + f_vjp((x1,[zero_term]))[0])

  # second pass
  yhs = jet(g, (x0,), ([x1,x2*fact(2)]+ [zero_term]*3,))

  f_jet = lambda x0: jet(g, (x0,), ((x1,x2*fact(2)),))
  yhs_wasted, f_vjp = vjp(f_jet, *(x0,))

  x3 = 1./3 * (yhs[1][1]/fact(2))
  x4 = 1./4 * (yhs[1][2]/fact(3) + f_vjp((x3,[zero_term,zero_term]))[0])
  x5 = 1./5 * (yhs[1][3]/fact(4) + f_vjp((x4,[x3,zero_term]))[0])
  x6 = 1./6 * (yhs[1][4]/fact(5) + f_vjp((x5,[x4,x3/fact(2)]))[0])

  return [x1, x2*fact(2), x3*fact(3), x4*fact(4), x5*fact(5), x6*fact(6)]

def vector_newton2(g,x0):
  zero_term = np.zeros_like(x0)
  one_term = np.ones_like(x0)

  # First Pass
  f_jet0 = lambda x0 : jet(g, (x0,), ((zero_term,),))
  yhs, f_jvp = linearize(f_jet0, x0)

  y0 = yhs[0]
  x1 = y0

  y1 = (yhs[1][0] + f_jvp(x1)[0])
  x2 = y1
  

  # second pass
  f_jet = lambda x0,x1,x2 : jet(g, (x0,), ([x1,x2] + [zero_term]*3,))
  yhs, f_jvp = linearize(f_jet,x0,x1,x2)

  y2 = yhs[1][1]
  x3 = y2
  print x3

  y3 = yhs[1][2] + fact(3) * f_jvp(zero_term,zero_term,x3/fact(3))[1][1]
  x4 = y3
  print x4

  y4 = yhs[1][3] + fact(4)/fact(2)*f_jvp(zero_term,x3*fact(1)/fact(3), x4*fact(2)/fact(4))[1][1]
  x5 = y4
  print x5

  y5 = yhs[1][4] + fact(5)/fact(2) *  f_jvp(x3 * fact(0)/fact(3), x4 * fact(1)/fact(4), x5 * fact(2)/fact(5))[1][1]
  x6 = y5
  print x6





  A0 = lambda v: jvp(lambda x0: jet(g, (x0, ), ((np.zeros_like(x1), ), )), (x0, ), ((v, )))[
      1][0]
  #dy1/dx0*v
  A1 = lambda v: jvp(lambda x0: jet(g, (x0, ), ((x1, ), )), (x0, ), ((v, )))[
      1][1][0]
