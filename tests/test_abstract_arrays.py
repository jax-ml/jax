import pytest
import numpy as np
from jax._src import abstract_arrays
from jax._src import core
from jax._src import dtypes

def test_abstract_array_conversions():
    """
    Test conversion mappings in abstract_arrays:
    - Verifies that a numpy ndarray is correctly converted to a ShapedArray with the same shape and canonicalized dtype.
    - Checks that a numpy scalar is converted to a ShapedArray with shape ().
    - Ensures a python scalar (int) conversion produces a ShapedArray with shape (), the expected dtype and weak_type flag.
    - Confirms that converting a numpy masked array raises a ValueError.
    """
    # Test numpy ndarray conversion
    np_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
    mapping_array = core.pytype_aval_mappings[np.ndarray]
    shaped_arr = mapping_array(np_array)
    assert shaped_arr.shape == np_array.shape, "The shaped array shape must match the numpy array shape."
    assert shaped_arr.dtype == dtypes.canonicalize_dtype(np_array.dtype), (
        "The shaped array dtype must be the canonicalized dtype of the numpy array."
    )
    
    # Test numpy scalar conversion
    np_scalar = np.int32(7)
    mapping_np_scalar = core.pytype_aval_mappings[type(np_scalar)]
    shaped_np_scalar = mapping_np_scalar(np_scalar)
    assert shaped_np_scalar.shape == (), "A numpy scalar should produce a shaped array with an empty shape ()."
    expected_np_dtype = dtypes.canonicalize_dtype(np.dtype(np_scalar).type)
    assert shaped_np_scalar.dtype == expected_np_dtype, "The dtype of the numpy scalar conversion is incorrect."
    
    # Test python scalar conversion
    py_int = 5
    mapping_py_int = core.pytype_aval_mappings[int]
    shaped_py_int = mapping_py_int(py_int)
    assert shaped_py_int.shape == (), "A python scalar should produce a shaped array with an empty shape ()."
    expected_py_dtype = dtypes._scalar_type_to_dtype(int, py_int)
    assert shaped_py_int.dtype == expected_py_dtype, "The dtype of the python int scalar conversion is incorrect."
    if int is not bool:
        assert shaped_py_int.weak_type, "Python int conversions should have weak_type set to True."
    
    # Test that numpy masked arrays are unsupported
    masked_arr = np.ma.masked_array([1, 2, 3])
    mapping_masked = core.pytype_aval_mappings[np.ma.MaskedArray]
    with pytest.raises(ValueError, match="numpy masked arrays are not supported as direct inputs to JAX functions."):
        mapping_masked(masked_arr)
def test_python_bool_conversion():
    """
    Test that a python bool conversion produces a ShapedArray with shape (),
    the expected canonicalized dtype, and that weak_type is False.
    """
    # Test python boolean conversion
    py_bool = True
    mapping_py_bool = core.pytype_aval_mappings[bool]
    shaped_py_bool = mapping_py_bool(py_bool)
    assert shaped_py_bool.shape == (), "A python bool should produce a shaped array with an empty shape ()."
    expected_bool_dtype = dtypes._scalar_type_to_dtype(bool, py_bool)
    assert shaped_py_bool.dtype == expected_bool_dtype, "The dtype of the python bool conversion is incorrect."
    assert not shaped_py_bool.weak_type, "Python bool conversions should not be weak (weak_type should be False)."
def test_python_float_conversion():
    """
    Test that a python float conversion produces a ShapedArray with an empty shape (),
    the expected canonical dtype for floats, and with weak_type set to True (since only bool is strong).
    """
    py_float = 3.14
    mapping_py_float = core.pytype_aval_mappings[float]
    shaped_py_float = mapping_py_float(py_float)
    assert shaped_py_float.shape == (), "A python float should produce a shaped array with shape ()."
    expected_dtype = dtypes._scalar_type_to_dtype(float, py_float)
    assert shaped_py_float.dtype == expected_dtype, "The dtype of the python float conversion is incorrect."
    assert shaped_py_float.weak_type, "Python float conversions should have weak_type set to True."
def test_python_complex_conversion():
    """
    Test that a python complex conversion produces a ShapedArray with an empty shape (),
    the expected canonicalized dtype using _scalar_type_to_dtype, and with weak_type set to True.
    """
    py_complex = 1 + 2j
    mapping_py_complex = core.pytype_aval_mappings[complex]
    shaped_py_complex = mapping_py_complex(py_complex)
    assert shaped_py_complex.shape == (), "A python complex number should produce a ShapedArray with shape ()."
    expected_complex_dtype = dtypes._scalar_type_to_dtype(complex, py_complex)
    assert shaped_py_complex.dtype == expected_complex_dtype, "The dtype of the python complex conversion is incorrect."
    assert shaped_py_complex.weak_type, "Python complex conversions should have weak_type set to True."
def test_literalable_types_contains_expected_types():
    """
    Test that core.literalable_types contains all expected types:
    - Verifies that it includes np.ndarray and all numpy scalar types (which together form array_types).
    - Verifies that it includes all python scalar types from dtypes.python_scalar_dtypes.
    """
    # Build the expected set of types.
    expected = set(abstract_arrays.array_types)  # includes np.ndarray and all numpy scalar types.
    expected.update(dtypes.python_scalar_dtypes.keys())  # add all python scalar types.
    # Check that every expected type is in core.literalable_types.
    missing = expected - core.literalable_types
    assert not missing, f"The core.literalable_types set is missing expected types: {missing}"
def test_invalid_numpy_array_dtype_conversion():
    """
    Test that converting a numpy array with an unsupported dtype (e.g., a string)
    raises an error. This ensures that dtypes.check_valid_dtype in
    _make_shaped_array_for_numpy_array properly rejects non-numeric dtypes.
    """
    # Create a numpy array with a non-numeric (string) dtype.
    invalid_array = np.array(["a", "b", "c"], dtype="<U1")
    mapping_array = core.pytype_aval_mappings[np.ndarray]
    # We expect that converting such an array raises an exception.
    # Depending on dtypes.check_valid_dtype, the exception may be a ValueError or TypeError.
    with pytest.raises((ValueError, TypeError)):
        mapping_array(invalid_array)
def test_numpy_zero_dim_array_conversion():
    """
    Test that a 0-dimensional numpy array (0-d array) is correctly converted to a ShapedArray with shape ().
    This ensures that arrays with no dimensions are handled properly.
    """
    # Create a 0-d numpy array.
    zero_d_array = np.array(42, dtype=np.int64)
    mapping_array = core.pytype_aval_mappings[np.ndarray]
    shaped_zero_d = mapping_array(zero_d_array)
    # Verify that the shape of the converted array is ()
    assert shaped_zero_d.shape == (), "A 0-d numpy array should produce a shaped array with an empty shape ()."
    # Verify that the dtype is the canonicalized dtype of the original array's dtype.
    expected_dtype = dtypes.canonicalize_dtype(zero_d_array.dtype)
    assert shaped_zero_d.dtype == expected_dtype, "The dtype of the 0-d numpy array conversion is incorrect."
def test_canonicalize_shape():
    """
    Test that the canonicalize_shape function correctly converts various shape 
    representations (e.g., list, tuple) into a canonical tuple.
    """
    # Test with a list shape.
    shape_list = [3, 4]
    canon_shape_list = abstract_arrays.canonicalize_shape(shape_list)
    assert isinstance(canon_shape_list, tuple), "canonicalize_shape should return a tuple."
    assert canon_shape_list == (3, 4), "The canonical shape for [3, 4] should be (3, 4)."
    # Test with a tuple shape (should remain unchanged).
    shape_tuple = (3, 4)
    canon_shape_tuple = abstract_arrays.canonicalize_shape(shape_tuple)
    assert isinstance(canon_shape_tuple, tuple), "canonicalize_shape should return a tuple."
    assert canon_shape_tuple == (3, 4), "The canonical shape for (3, 4) should be (3, 4)."
    # Test with an empty shape.
    empty_shape = []
    canon_empty = abstract_arrays.canonicalize_shape(empty_shape)
    assert canon_empty == (), "The canonical shape for an empty list should be ()."
    # Optionally, test with a 1-dimensional input.
    shape_1d = [5]
    canon_shape_1d = abstract_arrays.canonicalize_shape(shape_1d)
    assert canon_shape_1d == (5,), "The canonical shape for [5] should be (5,)."