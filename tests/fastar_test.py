import numpy as onp
from absl.testing import parameterized

import jax.lax as lax
import jax.scipy.special as special
from jax import numpy as np, jit
from jax import test_util as jtu
from jax.interpreters.fastar import accelerate_part, Parray
from jax.fastar_util import false_mask, mask_to_slices
from jax.util import safe_map, safe_zip

map = safe_map
zip = safe_zip

def increasing_masks(rng, *arrs_raw):
  idxs = shuffled_idxs(rng, *arrs_raw)
  masks = [false_mask(arr) for arr in arrs_raw]
  arrs = []
  for argnum, idx in idxs:
    mask = onp.copy(masks[argnum])
    mask[idx] = True
    masks[argnum] = mask
    arrs.append(tuple(Parray((arr * mask, mask))
                      for arr, mask in zip(arrs_raw, masks)))
  return arrs


def shuffled_idxs(rng, *arrs):
  idxs = sum(([(argnum, idx) for idx in onp.ndindex(np.shape(arr))]
              for argnum, arr in enumerate(arrs)), [])
  perm = rng.permutation(len(idxs))
  return [idxs[i] for i in perm]


def is_subset(mask_1, mask_2):
  return np.all(~mask_1 | mask_2)


def check_ans(ans_old, ans):
  ans_old, mask_old = ans_old
  ans, mask = ans
  assert isinstance(mask, (bool, np.bool_, np.ndarray))
  assert is_subset(mask_old, mask)
  assert is_subset(onp.bool_(ans), mask)
  assert np.all(np.where(mask_old, ans == ans_old, True))


def check_custom_input(fun, inputs_from_rng, rtol=1e-5, atol=1e-8, runs=2):
  rng = onp.random.RandomState(0)
  for _ in range(runs):
    args = inputs_from_rng(rng)
    ans = fun(*args)
    fun_ac = accelerate_part(fun, jit=False)
    masks = increasing_masks(rng, *args)
    args_ = [Parray((arg, false_mask(arg))) for arg in args]
    ans_old, fun_ac = fun_ac(*args_)
    for args in masks:
      ans_, fun_ac = fun_ac(*args)
      check_ans(ans_old, ans_)
      ans_old = ans_
    ans_, mask = ans_
    assert np.all(mask)
    try:
      assert np.allclose(ans, ans_, rtol=rtol, atol=atol)
    except AssertionError:
      np.set_printoptions(threshold=np.nan)
      raise AssertionError(
        'Result incorrect: ' + str(ans) + 'vs. \n' + str(ans_) +
        '\n: Differs at: ' + str(np.isclose(ans, ans_, rtol=rtol, atol=atol)) +
        '\n Difference: ' + str(ans - ans_))
    assert ans.dtype == ans_.dtype


def check(fun, *shapes, **kwargs):
  check_custom_input(fun, lambda rng: tuple(rng.randn(*shape)
                                            for shape in shapes), **kwargs)


class FastarTest(jtu.JaxTestCase):
  @parameterized.named_parameters(
    jtu.cases_from_list({'testcase_name': str(op), 'op': op}
                        for op in (lax.abs, lax.ceil, lax.cos, lax.sin, lax.exp, lax.expm1,
                                   lax.floor, lax.neg, lax.sign, lax.tanh, special.expit)))
  def test_unop(self, op):
    check(op, (1, 2))

  @parameterized.named_parameters(
    jtu.cases_from_list({'testcase_name': str(op), 'op': op} for op in (lax.log, lax.log1p)))
  def test_unop_positive_arg(self, op):
    check_custom_input(op, lambda rng: np.abs(rng.randn(1, 2)))

  @parameterized.named_parameters(jtu.cases_from_list({'testcase_name': str(dtype), 'new_dtype': dtype}
                                                      for dtype in (onp.int, onp.bool, onp.byte)))
  def test_convert(self, new_dtype):
    check(lambda x: lax.convert_element_type(x, new_dtype=new_dtype), (1, 2))

  def test_logit(self):
    check_custom_input(special.logit, lambda rng: rng.rand(1, 2))

  @parameterized.named_parameters(
    jtu.cases_from_list({'testcase_name': str(op), 'op': op}
                        for op in (lax.add, lax.div, lax.eq, lax.ge, lax.gt, lax.le, lax.lt,
                                   lax.max, lax.min, lax.mul, lax.ne, lax.rem, lax.sub)))
  def test_binop(self, op):
    check(op, (1, 2), (3, 1))

  def test_add_scalar_scalar(self):
    check(lax.add, (), ())

  def test_add_scalar_vector(self):
    check(lax.add, (), (2,))

  def test_add_scalar_scalar_as_numpy(self):
    check_custom_input(lax.add, lambda rng: (np.float64(rng.randn()), np.float64(rng.randn())))

  def test_add_scalar_scalar_as_array(self):
    check_custom_input(lax.add, lambda rng: (np.array(rng.randn()), np.array(rng.randn())))

  def test_add_scalar_scalar_int(self):
    check_custom_input(lax.add, lambda _: (4, 7))

  def test_add_vector_vector(self):
    check(lax.add, (1,), (2,))

  @parameterized.named_parameters({'testcase_name': "axes={}_shape={}".format(axes, shape),
                                   'axes': axes, 'shape': shape}
                                  for axes, shape in (((0,), (4,)),
                                                      ((0,), (2, 4)),
                                                      ((1,), (2, 4)),
                                                      ((0, 1), (2, 4)),
                                                      ((0, 2), (2, 3, 4))))
  def test_reduce_sum(self, axes, shape):
    check(lambda x: lax._reduce_sum(x, axes=axes), shape)

  def test_reduce_min(self):
    check(lambda x: lax._reduce_min(x, axes=(0,)), (2, 4))

  def test_reduce_max(self):
    check(lambda x: lax._reduce_max(x, axes=(0,)), (2, 4))

  @parameterized.named_parameters({'testcase_name': str(shapes), 'shapes': shapes}
                                  for shapes in (((2,), (2,)),
                                                 ((3, 2), (2,)),
                                                 ((2,), (2, 3)),
                                                 ((3, 2), (2, 4))))
  def test_dot_vector_vector(self, shapes):
    check(lax.dot, *shapes)

  def test_dot_general_tensor_matrix(self):
    check(lambda x, y: lax.dot_general(x, y, dimension_numbers=(((2,), (1,)), ((0,), (0,)))), (1, 2, 3), (1, 3))

  def test_dot_general_tensor_tensor(self):
    check(lambda x, y: lax.dot_general(x, y, dimension_numbers=(((2,), (2,)), ((1, 0), (1, 0)))),
          (1, 2, 3), (1, 2, 3, 4), rtol=1e-4)

  def test_select(self):
    check_custom_input(lax.select, lambda rng: (rng.rand(3, 2) > 0.5, rng.randn(3, 2), rng.randn(3, 2)))

  def test_select_scalar_pred(self):
    check_custom_input(lax.select, lambda rng: (rng.rand() > 0.5, rng.randn(3, 2), rng.randn(3, 2)))

  def test_transpose(self):
    check(lambda x: lax.transpose(x, permutation=(1, 2, 0)), (1, 2, 3))

  def test_reverse(self):
    check(lambda x: lax.rev(x, dimensions=(1, 2)), (1, 2, 3))

  def test_reshape(self):
    check(lambda x: lax.reshape(x, new_sizes=(3, 2), dimensions=(1, 0, 2)), (1, 2, 3))

  def test_concatenate_2(self):
    check(lambda x, y: lax.concatenate((x, y), dimension=2), (1, 2, 1), (1, 2, 3))

  def test_concatenate_3(self):
    check(lambda x, y, z: lax.concatenate((x, y, z), dimension=1), (2, 1), (2, 3), (2, 2))

  def test_slice(self):
    check(lambda x: lax.slice(x, start_indices=(1, 1, 0), limit_indices=(2, 2, 1), strides=(2, 1, 1)), (7, 2, 1))

  # TODO implement simultaneous negative + interior:
  # (((-1, -2, 2),), ((5,), ())), (((-1, -2, 1), (1, 2, 2)), ((4, 2), ()))
  @parameterized.named_parameters({'testcase_name': "padding_config={}_shapes={}".format(padding_config, shapes),
                                   'padding_config': padding_config, 'shapes': shapes}
                                  for padding_config, shapes in (
                                          (((1, 2, 0),), ((2,), ())),
                                          (((1, 2, 0), (3, 4, 0)), ((1, 2), ())),
                                          (((0, 0, 0), (0, 0, 0)), ((1, 2), ())),
                                          (((1, 2, 3),), ((2,), ())),
                                          (((1, 2, 1), (3, 4, 2)), ((3, 2), ())),
                                          (((-1, 2, 0),), ((2,), ())),
                                          (((-1, -2, 0), (1, 2, 0)), ((4, 2), ())),
                                          (((-1, 2, 0), (1, 2, 2)), ((4, 2), ()))))
  def test_pad_vector(self, padding_config, shapes):
    check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=padding_config), *shapes)

  @parameterized.named_parameters(jtu.cases_from_list(
    {'testcase_name': "strides={}_padding={}_lhs_dilation={}_dimension_numbers={}_lhs_perm={}, rhs_perm={}".format(
      strides, padding, lhs_dilation, dimension_numbers, lhs_perm, rhs_perm),
      'strides': strides, 'padding': padding, 'lhs_dilation': lhs_dilation,
      'dimension_numbers': dimension_numbers, 'lhs_perm': lhs_perm, 'rhs_perm': rhs_perm}
    for strides in ((1, 2), (2, 1))
    for padding in (((0, 0), (0, 0)), 'VALID', 'SAME')
    for lhs_dilation in (None, (1, 2))
    for dimension_numbers, (lhs_perm, rhs_perm) in (
            (("NCHW", "OIHW", "NCHW"), ((0, 1, 2, 3), (0, 1, 2, 3))),
            (("NHWC", "HWIO", "NHWC"), ((0, 2, 3, 1), (2, 3, 1, 0))),
            (("NCHW", "HWIO", "NHWC"), ((0, 1, 2, 3), (2, 3, 1, 0))))))
  def test_convolution(self, strides, padding, lhs_dilation, dimension_numbers, lhs_perm, rhs_perm):
    if lhs_dilation is not None and isinstance(padding, str):
      return # String padding is not implemented for transposed convolution, see conv_general_dilated implementation

    lhs_shape = (1, 2, 4, 4)
    rhs_shape = (3, 2, 3, 2)
    check(
      lambda lhs: lax.conv_general_dilated(
        lhs, onp.random.RandomState(0).randn(*np.take(rhs_shape, rhs_perm)),
        strides, padding, lhs_dilation=lhs_dilation,
        dimension_numbers=dimension_numbers),
      np.take(lhs_shape, lhs_perm), rtol=1e-4, atol=1e-6)

  def test_jit(self):
    check(jit(lambda x: x * 2), (1,))

  def test_jit_freevar(self):
    check(lambda x, y: jit(lambda x: x * y)(x), (1,), (1,))

  def test_mask_to_slices(self):
    assert mask_to_slices(False) == []
    assert mask_to_slices(True) == [()]
    assert mask_to_slices(onp.array(False)) == []
    assert mask_to_slices(onp.array(True)) == [()]
    assert mask_to_slices(onp.array([True, False, False, True, True])) == [
      (slice(0, 1, None),), (slice(3, 5, None),)]
    assert mask_to_slices(onp.array([
      [True, True, False],
      [True, True, True],
      [True, True, False]])) == [
             (slice(0, 3, None), slice(0, 2, None)),
             (slice(1, 2, None), slice(2, 3, None))]
