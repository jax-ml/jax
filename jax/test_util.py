# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import re
import itertools as it
import random

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import numpy.random as npr

from six.moves import xrange

from . import api
from .config import flags
from .util import partial
from .tree_util import tree_multimap, tree_all, tree_map, tree_reduce

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    'jax_test_dut',
    None,
    enum_values=['cpu', 'gpu', 'tpu'],
    help=
    'Describes the device under test in case special consideration is required.'
)

flags.DEFINE_integer(
  'num_generated_cases',
  100,
  help='Number of generated cases to test')

EPS = 1e-4
ATOL = 1e-4
RTOL = 1e-4

_dtype = lambda x: getattr(x, 'dtype', None) or onp.asarray(x).dtype


def numpy_eq(x, y):
  testing_tpu = FLAGS.jax_test_dut and FLAGS.jax_test_dut.startswith("tpu")
  testing_x32 = not FLAGS.jax_enable_x64
  if testing_tpu or testing_x32:
    return onp.allclose(x, y, 1e-3, 1e-3)
  else:
    return onp.allclose(x, y)


def numpy_close(a, b, atol=ATOL, rtol=RTOL, equal_nan=False):
  testing_tpu = FLAGS.jax_test_dut and FLAGS.jax_test_dut.startswith("tpu")
  testing_x32 = not FLAGS.jax_enable_x64
  if testing_tpu or testing_x32:
    atol = max(atol, 1e-1)
    rtol = max(rtol, 1e-1)
  return onp.allclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def check_eq(xs, ys):
  assert tree_all(tree_multimap(numpy_eq, xs, ys)), \
      '\n{} != \n{}'.format(xs, ys)


def check_close(xs, ys, atol=ATOL, rtol=RTOL):
  close = partial(numpy_close, atol=atol, rtol=rtol)
  assert tree_all(tree_multimap(close, xs, ys)), '\n{} != \n{}'.format(xs, ys)


def inner_prod(xs, ys):
  contract = lambda x, y: onp.real(onp.vdot(x, y))
  return tree_reduce(onp.add, tree_multimap(contract, xs, ys))


add = partial(tree_multimap, onp.add)
sub = partial(tree_multimap, onp.subtract)
conj = partial(tree_map, onp.conj)


def scalar_mul(xs, a):
  return tree_map(lambda x: onp.multiply(x, a, dtype=_dtype(x)), xs)


def rand_like(rng, x):
  shape = onp.shape(x)
  dtype = _dtype(x)
  randn = lambda: onp.asarray(rng.randn(*shape), dtype=dtype)
  if onp.issubdtype(dtype, onp.complexfloating):
    return randn() + 1.0j * randn()
  else:
    return randn()


def numerical_jvp(f, primals, tangents, eps=EPS):
  delta = scalar_mul(tangents, EPS)
  f_pos = f(*add(primals, delta))
  f_neg = f(*sub(primals, delta))
  return scalar_mul(sub(f_pos, f_neg), 0.5 / EPS)


def check_jvp(f, f_jvp, args, atol=ATOL, rtol=RTOL, eps=EPS):
  rng = onp.random.RandomState(0)
  tangent = tree_map(partial(rand_like, rng), args)
  v_out, t_out = f_jvp(args, tangent)
  v_out_expected = f(*args)
  t_out_expected = numerical_jvp(f, args, tangent, eps=eps)
  check_eq(v_out, v_out_expected)
  check_close(t_out, t_out_expected, atol=atol, rtol=rtol)


def check_vjp(f, f_vjp, args, atol=ATOL, rtol=RTOL, eps=EPS):
  _rand_like = partial(rand_like, onp.random.RandomState(0))
  v_out, vjpfun = f_vjp(*args)
  v_out_expected = f(*args)
  check_eq(v_out, v_out_expected)
  tangent = tree_map(_rand_like, args)
  tangent_out = numerical_jvp(f, args, tangent, eps=EPS)
  cotangent = tree_map(_rand_like, v_out)
  cotangent_out = conj(vjpfun(conj(cotangent)))
  ip = inner_prod(tangent, cotangent_out)
  ip_expected = inner_prod(tangent_out, cotangent)
  check_close(ip, ip_expected, atol=atol, rtol=rtol)


def skip_on_devices(*disabled_devices):
  """A decorator for test methods to skip the test on certain devices."""
  def skip(test_method):
    @functools.wraps(test_method)
    def test_method_wrapper(self, *args, **kwargs):
      device = FLAGS.jax_test_dut
      if device in disabled_devices:
        test_name = getattr(test_method, '__name__', '[unknown test]')
        return absltest.unittest.skip(
            '{} not supported on {}.'.format(test_name, device.upper()))
      return test_method(self, *args, **kwargs)
    return test_method_wrapper
  return skip


def skip_on_flag(flag_name, skip_value):
  """A decorator for test methods to skip the test when flags are set."""
  def skip(test_method):        # pylint: disable=missing-docstring
    @functools.wraps(test_method)
    def test_method_wrapper(self, *args, **kwargs):
      flag_value = getattr(FLAGS, flag_name)
      if flag_value == skip_value:
        test_name = getattr(test_method, '__name__', '[unknown test]')
        return absltest.unittest.skip(
            '{} not supported when FLAGS.{} is {}'.format(
                test_name, flag_name, flag_value))
      return test_method(self, *args, **kwargs)
    return test_method_wrapper
  return skip


def format_test_name_suffix(opname, shapes, dtypes):
  arg_descriptions = (format_shape_dtype_string(shape, dtype)
                      for shape, dtype in zip(shapes, dtypes))
  return '{}_{}'.format(opname.capitalize(), '_'.join(arg_descriptions))


class _NumpyScalar(object):

  def __len__(self):
    return 0

# A special singleton "shape" that denotes numpy scalars. Numpy scalars are not
# identical to 0-D arrays, and we want to write tests that exercise both paths.
NUMPY_SCALAR_SHAPE = _NumpyScalar()


def _dims_of_shape(shape):
  """Converts `shape` to a tuple of dimensions."""
  return shape if shape != NUMPY_SCALAR_SHAPE else ()


def _cast_to_shape(value, shape, dtype):
  """Casts `value` to the correct Python type for `shape` and `dtype`."""
  if shape != NUMPY_SCALAR_SHAPE:
    return value
  else:
    # A numpy scalar was requested. Explicitly cast in case `value` is a Python
    # scalar.
    return dtype(value)


def format_shape_dtype_string(shape, dtype):
  typestr = onp.dtype(dtype).name
  if shape == NUMPY_SCALAR_SHAPE:
    return typestr

  if onp.isscalar(shape):
    shapestr = str(shape) + ','
  else:
    shapestr = ','.join(str(dim) for dim in shape)
  return '{}[{}]'.format(typestr, shapestr)


def _rand_dtype(rand, shape, dtype, scale=1., post=lambda x: x):
  """Produce random values given shape, dtype, scale, and post-processor.

  Args:
    rand: a function for producing random values of a given shape, e.g. a
      bound version of either onp.RandomState.randn or onp.RandomState.rand.
    shape: a shape value as a tuple of positive integers.
    dtype: a numpy dtype.
    scale: optional, a multiplicative scale for the random values (default 1).
    post: optional, a callable for post-processing the random values (default
      identity).

  Returns:
    An ndarray of the given shape and dtype using random values based on a call
    to rand but scaled, converted to the appropriate dtype, and post-processed.
  """
  r = lambda: onp.asarray(scale * rand(*_dims_of_shape(shape)), dtype)
  if onp.issubdtype(dtype, onp.complexfloating):
    vals = r() + 1.0j * r()
  else:
    vals = r()
  return _cast_to_shape(onp.asarray(post(vals), dtype), shape, dtype)


def rand_default():
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=3)


def rand_nonzero():
  post = lambda x: onp.where(x == 0, 1, x)
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=3, post=post)


def rand_positive():
  post = lambda x: x + 1
  rand = npr.RandomState(0).rand
  return partial(_rand_dtype, rand, scale=2, post=post)


def rand_small():
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=1e-3)


def rand_not_small():
  post = lambda x: x + onp.where(x > 0, 10., -10.)
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=3., post=post)


def rand_small_positive():
  rand = npr.RandomState(0).rand
  return partial(_rand_dtype, rand, scale=2e-5)


def rand_some_equal():
  randn = npr.RandomState(0).randn
  rng = npr.RandomState(0)

  def post(x):
    flips = rng.rand(*onp.shape(x)) < 0.5
    return onp.where(flips, x.ravel()[0], x)

  return partial(_rand_dtype, randn, scale=100., post=post)


# TODO(mattjj): doesn't handle complex types
def rand_some_inf():
  """Return a random sampler that produces infinities in floating types."""
  rng = npr.RandomState(1)
  base_rand = rand_default()

  def rand(shape, dtype):
    """The random sampler function."""
    if not onp.issubdtype(dtype, onp.float):
      # only float types have inf
      return base_rand(shape, dtype)

    dims = _dims_of_shape(shape)
    posinf_flips = rng.rand(*dims) < 0.1
    neginf_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = onp.where(posinf_flips, onp.inf, vals)
    vals = onp.where(neginf_flips, -onp.inf, vals)

    return _cast_to_shape(onp.asarray(vals, dtype=dtype), shape, dtype)

  return rand


# TODO(mattjj): doesn't handle complex types
def rand_some_zero():
  """Return a random sampler that produces some zeros."""
  rng = npr.RandomState(1)
  base_rand = rand_default()

  def rand(shape, dtype):
    """The random sampler function."""
    dims = _dims_of_shape(shape)
    zeros = rng.rand(*dims) < 0.5

    vals = base_rand(shape, dtype)
    vals = onp.where(zeros, 0, vals)

    return _cast_to_shape(onp.asarray(vals, dtype=dtype), shape, dtype)

  return rand


def rand_bool():
  rng = npr.RandomState(0)
  def generator(shape, dtype):
    return _cast_to_shape(rng.rand(*_dims_of_shape(shape)) < 0.5, shape, dtype)
  return generator

def check_raises(thunk, err_type, msg):
  try:
    thunk()
    assert False
  except err_type as e:
    assert str(e).startswith(msg), "\n{}\n\n{}\n".format(e, msg)

def check_raises_regexp(thunk, err_type, pattern):
  try:
    thunk()
    assert False
  except err_type as e:
    assert re.match(pattern, str(e)), "{}\n\n{}\n".format(e, pattern)

random.seed(0) # TODO: consider managing prng state more carefully

def cases_from_list(xs):
  xs = list(xs)
  k = min(len(xs), FLAGS.num_generated_cases)
  return random.sample(xs, k)

def cases_from_gens(*gens):
  sizes = [1, 3, 10]
  cases_per_size = int(FLAGS.num_generated_cases / len(sizes)) + 1
  for size in sizes:
    for i in xrange(cases_per_size):
      yield ('_{}_{}'.format(size, i),) + tuple(gen(size) for gen in gens)


class JaxTestCase(parameterized.TestCase):
  """Base class for JAX tests including numerical checks and boilerplate."""

  def assertArraysAllClose(self, x, y, check_dtypes, atol=None, rtol=None):
    """Assert that x and y are close (up to numerical tolerances)."""
    dtype = lambda x: str(onp.asarray(x).dtype)
    tol = 1e-2 if str(onp.dtype(onp.float32)) in {dtype(x), dtype(y)} else 1e-5
    atol = atol or tol
    rtol = rtol or tol

    if FLAGS.jax_test_dut == 'tpu':
      atol = max(atol, 0.5)
      rtol = max(rtol, 1e-1)

    if not onp.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True):
      msg = ('Arguments x and y not equal to tolerance atol={}, rtol={}:\n'
             'x:\n{}\n'
             'y:\n{}\n').format(atol, rtol, x, y)
      raise self.failureException(msg)

    if check_dtypes:
      self.assertDtypesMatch(x, y)

  def assertDtypesMatch(self, x, y):
    if FLAGS.jax_enable_x64:
      self.assertEqual(onp.asarray(x).dtype, onp.asarray(y).dtype)

  def assertAllClose(self, x, y, check_dtypes, atol=None, rtol=None):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, (tuple, list)):
      self.assertIsInstance(y, (tuple, list))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self.assertAllClose(x_elt, y_elt, check_dtypes, atol=atol, rtol=rtol)
    else:
      is_array = lambda x: hasattr(x, '__array__') or onp.isscalar(x)
      self.assertTrue(is_array(x))
      self.assertTrue(is_array(y))
      x = onp.asarray(x)
      y = onp.asarray(y)
      self.assertArraysAllClose(x, y, check_dtypes, atol=atol, rtol=rtol)

  def _CompileAndCheck(self, fun, args_maker, check_dtypes,
                       rtol=None, atol=None):
    """Helper method for running JAX compilation and allclose assertions."""
    args = args_maker()

    def wrapped_fun(*args):
      self.assertTrue(python_should_be_executing)
      return fun(*args)

    python_should_be_executing = True
    python_ans = fun(*args)

    cfun = api.jit(wrapped_fun)
    python_should_be_executing = True
    monitored_ans = cfun(*args)

    python_should_be_executing = False
    compiled_ans = cfun(*args)

    self.assertAllClose(python_ans, monitored_ans, check_dtypes, rtol, atol)
    self.assertAllClose(python_ans, compiled_ans, check_dtypes, rtol, atol)

    args = args_maker()

    python_should_be_executing = True
    python_ans = fun(*args)

    python_should_be_executing = False
    compiled_ans = cfun(*args)

    self.assertAllClose(python_ans, compiled_ans, check_dtypes, rtol, atol)

  def _CheckAgainstNumpy(self, lax_op, numpy_reference_op, args_maker,
                         check_dtypes=False, tol=1e-5):
    args = args_maker()
    lax_ans = lax_op(*args)
    numpy_ans = numpy_reference_op(*args)
    self.assertAllClose(lax_ans, numpy_ans, check_dtypes=check_dtypes,
                        atol=tol, rtol=tol)
