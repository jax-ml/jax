import pytest
import jax.numpy as jnp
from jax._src.experimental.linalg.linear_operators import (
    LinearOperator,
    IdentityOperator,
    ScalingOperator,
    MatrixOperator,
    CompositionOperator,
    DiagonalOperator,
    TransposeOperator,
)



# ---------------------------------------EASY TEST CASES------------------------------------------ 

# Basic IdentityOperator tests
def test_identity_operator_returns_input():
    x = jnp.array([1.0, 2.0, 3.0])
    I = IdentityOperator(size=3)
    y = I(x)
    assert jnp.allclose(y, x)


# ScalingOperator tests
def test_scaling_operator_scales_correctly():
    x = jnp.array([1.0, -2.0, 3.0])
    S = ScalingOperator(scale=2.0, size=3)
    y = S(x)
    assert jnp.allclose(y, 2.0 * x)



# MatrixOperator tests
def test_matrix_operator_matches_explicit_dot():
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    x = jnp.array([1.0, -1.0])
    M = MatrixOperator(A)
    y = M(x)
    assert jnp.allclose(y, jnp.dot(A, x))



# CompositionOperator tests
def test_composition_operator_matches_manual_composition():
    A = MatrixOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    B = ScalingOperator(scale=2.0, size=2)
    C = CompositionOperator(A, B)
    x = jnp.array([1.0, -1.0])
    y = C(x)
    expected = A(B(x))
    assert jnp.allclose(y, expected)


# DiagonalOperator tests
def test_diagonal_operator_multiplies_elementwise():
    diag = jnp.array([1.0, 2.0, 3.0])
    D = DiagonalOperator(diag)
    x = jnp.array([2.0, 2.0, 2.0])
    y = D(x)
    assert jnp.allclose(y, diag * x)


# TransposeOperator tests
def test_transpose_operator_matches_matrix_transpose():
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    M = MatrixOperator(A)
    T = TransposeOperator(M)
    x = jnp.array([1.0, -1.0])
    y = T(x)
    expected = jnp.dot(A.T, x)
    assert jnp.allclose(y, expected)


# to_dense() and operator algebra tests
def test_to_dense_recreates_matrix_operator():
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    M = MatrixOperator(A)
    dense = M.to_dense()
    assert jnp.allclose(dense, A)


def test_operator_addition_and_scalar_multiplication():
    A = MatrixOperator(jnp.array([[1.0, 0.0], [0.0, 2.0]]))
    B = ScalingOperator(scale=1.0, size=2)
    x = jnp.array([3.0, 4.0])

    sum_op = A + B
    scaled_op = 2.0 * A

    y_sum = sum_op(x)
    y_scaled = scaled_op(x)

    expected_sum = A(x) + B(x)
    expected_scaled = 2.0 * A(x)

    assert jnp.allclose(y_sum, expected_sum)
    assert jnp.allclose(y_scaled, expected_scaled)



# ---------------------------------------MEDIUM TEST CASES------------------------------------------ 

def test_composed_operator_with_diagonal_and_scaling():
    """Test composition of Diagonal and Scaling operators"""
    diag = jnp.array([1.0, 3.0, 5.0])
    D = DiagonalOperator(diag)
    S = ScalingOperator(2.0, size=3)
    C = CompositionOperator(D, S)  # D(S(x)) = 2 * D(x)
    x = jnp.array([1.0, 1.0, 1.0])
    y = C(x)
    expected = 2.0 * diag * x
    assert jnp.allclose(y, expected)


def test_addition_and_scaling_chains_correctly():
    """Check that scalar and operator addition behave as expected"""
    A = MatrixOperator(jnp.array([[1.0, 0.0], [0.0, 2.0]]))
    B = MatrixOperator(jnp.array([[0.5, 0.5], [0.5, 0.5]]))
    x = jnp.array([2.0, 4.0])

    combo = 2.0 * A + B
    y = combo(x)
    expected = 2.0 * (A(x)) + B(x)
    assert jnp.allclose(y, expected)


def test_transpose_of_composed_operator_matches_dense():
    """Test (A(B(x)))^T ≈ (B^T A^T)(x) numerically"""
    A = MatrixOperator(jnp.array([[1., 2.], [3., 4.]]))
    B = MatrixOperator(jnp.array([[0., 1.], [1., 0.]]))  # swap operator
    C = CompositionOperator(A, B)
    T = TransposeOperator(C)
    x = jnp.array([1.0, -1.0])
    y = T(x)
    expected = jnp.dot(C.to_dense().T, x)
    assert jnp.allclose(y, expected)



# ---------------------------------------HARD TEST CASES------------------------------------------ 

def test_shape_mismatch_raises_assertion():
    """Ensure shape mismatch produces a clear assertion"""
    A = MatrixOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    x = jnp.array([1.0, 2.0, 3.0])  # wrong shape
    with pytest.raises(AssertionError):
        _ = A(x)


def test_large_random_operator_consistency():
    """For a large operator, to_dense() and matvec() agree"""
    key = jnp.arange(0, 100).reshape(10, 10) * 0.01
    A = MatrixOperator(key)
    x = jnp.linspace(-1, 1, 10)
    dense = A.to_dense()
    y_dense = jnp.dot(dense, x)
    y_op = A(x)
    assert jnp.allclose(y_dense, y_op, atol=1e-6)


def test_combined_linear_expression_equivalence():
    """Test ((A + B) @ x) == A@x + B@x for random matrices"""
    A = MatrixOperator(jnp.array([[1., 2.], [0., 1.]]))
    B = MatrixOperator(jnp.array([[0., 1.], [3., -1.]]))
    x = jnp.array([0.5, -0.5])
    sum_op = A + B
    y_sum = sum_op(x)
    expected = A(x) + B(x)
    assert jnp.allclose(y_sum, expected)




# ---------------------------------------UGLY EDGE TEST CASES------------------------------------------ 

def test_scaling_operator_zero_and_negative():
    """Scaling by 0 or negative values behaves correctly"""
    x = jnp.array([1.0, -2.0, 3.0])
    zero_op = ScalingOperator(scale=0.0, size=3)
    neg_op = ScalingOperator(scale=-1.0, size=3)

    y_zero = zero_op(x)
    y_neg = neg_op(x)

    assert jnp.allclose(y_zero, jnp.zeros_like(x))
    assert jnp.allclose(y_neg, -x)


def test_empty_operator_behavior():
    """Ensure 0x0 operators don't crash"""
    A = MatrixOperator(jnp.zeros((0, 0)))
    x = jnp.zeros((0,))
    y = A(x)
    assert y.shape == (0,)


def test_nested_composition_three_layers():
    """Nested composition should behave correctly"""
    A = MatrixOperator(jnp.array([[1., 2.], [3., 4.]]))
    B = ScalingOperator(0.5, 2)
    C = DiagonalOperator(jnp.array([2., 3.]))

    composed = CompositionOperator(A, CompositionOperator(B, C))  # A(B(C(x)))
    x = jnp.array([1.0, -1.0])
    y = composed(x)

    expected = A(B(C(x)))
    assert jnp.allclose(y, expected)



