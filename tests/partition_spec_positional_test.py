import numpy as np
import pytest

from jax.sharding import PartitionSpec as P, Mesh


# ============================================================================
# BASIC POSITIVE INDEX TESTS
# ============================================================================

def test_p0_expands_1d():
    """Test that P(0) expands to P("i") with a 1D mesh."""
    mesh = Mesh(np.empty((4,), dtype=object), ("i",))
    assert P(0)._expand_with_mesh(mesh) == P("i")


def test_p01_expands_2d():
    """Test that P(0, 1) expands to P("i", "j") with a 2D mesh."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    assert P(0, 1)._expand_with_mesh(mesh) == P("i", "j")


def test_p012_expands_3d():
    """Test that P(0, 1, 2) expands correctly with a 3D mesh."""
    mesh = Mesh(np.empty((2, 2, 2), dtype=object), ("i", "j", "k"))
    assert P(0, 1, 2)._expand_with_mesh(mesh) == P("i", "j", "k")


def test_p0_with_none():
    """Test that P(None, 0) expands correctly."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    assert P(None, 0)._expand_with_mesh(mesh) == P(None, "i")


def test_p1_with_none():
    """Test that P(None, 1) expands correctly."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    assert P(None, 1)._expand_with_mesh(mesh) == P(None, "j")


# ============================================================================
# NEGATIVE INDEX TESTS (excluding -1 standalone)
# ============================================================================

def test_p_minus2():
    """Test that P(-2) expands to the second-to-last axis."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    assert P(-2)._expand_with_mesh(mesh) == P("i")


def test_p_minus1_alone_1d():
    """Test that P(-1) expands to the last axis in a 1D mesh."""
    mesh1 = Mesh(np.empty((4,), dtype=object), ("i",))
    assert P(-1)._expand_with_mesh(mesh1) == P("i")


def test_p_minus1_alone_2d():
    """Test that P(-1) expands to all axes in a 2D mesh."""
    mesh2 = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    assert P(-1)._expand_with_mesh(mesh2) == P(("i", "j"))


def test_p_minus1_alone_3d():
    """Test that P(-1) expands to all axes in a 3D mesh."""
    mesh3 = Mesh(np.empty((2, 2, 2), dtype=object), ("i", "j", "k"))
    assert P(-1)._expand_with_mesh(mesh3) == P(("i", "j", "k"))


def test_p_minus2_minus1():
    """Test that P(-2, -1) expands correctly in a 3D mesh."""
    mesh = Mesh(np.empty((2, 2, 2), dtype=object), ("i", "j", "k"))
    assert P(-2, -1)._expand_with_mesh(mesh) == P("j", "k")


# ============================================================================
# MIXED POSITIVE/NEGATIVE AND WILDCARD (-1) TESTS
# ============================================================================

def test_mixed_p0_minus1():
    """Test P(0, -1) expands to P("i", "j") in a 2D mesh."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    assert P(0, -1)._expand_with_mesh(mesh) == P("i", "j")


def test_mixed_p0_minus1_3d():
    """Test P(0, -1) expands to P("i", ("j", "k")) in a 3D mesh."""
    mesh = Mesh(np.empty((2, 2, 2), dtype=object), ("i", "j", "k"))
    assert P(0, -1)._expand_with_mesh(mesh) == P("i", ("j", "k"))


def test_mixed_p1_minus1_3d():
    """Test P(1, -1) expands correctly in a 3D mesh."""
    mesh = Mesh(np.empty((2, 2, 2), dtype=object), ("i", "j", "k"))
    assert P(1, -1)._expand_with_mesh(mesh) == P("j", ("i", "k"))


def test_p_none_minus1():
    """Test P(None, -1) expands to P(None, ("i", "j")) in a 2D mesh."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    assert P(None, -1)._expand_with_mesh(mesh) == P(None, ("i", "j"))


def test_p_none_0_minus1():
    """Test P(None, 0, -1) expands correctly."""
    mesh = Mesh(np.empty((2, 2, 2), dtype=object), ("i", "j", "k"))
    assert P(None, 0, -1)._expand_with_mesh(mesh) == P(None, "i", ("j", "k"))


# ============================================================================
# TUPLE LEAF TESTS
# ============================================================================

def test_tuple_leaf_01():
    """Test that P((0, 1)) expands to P(("i", "j"))."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    assert P((0, 1))._expand_with_mesh(mesh) == P(("i", "j"))


def test_tuple_leaf_0_minus1():
    """Test that P((0, -1)) expands correctly."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    assert P((0, -1))._expand_with_mesh(mesh) == P(("i", "j"))


def test_tuple_leaf_0_minus1_3d():
    """Test that P((0, -1)) in a 3D mesh expands correctly."""
    mesh = Mesh(np.empty((2, 2, 2), dtype=object), ("i", "j", "k"))
    assert P((0, -1))._expand_with_mesh(mesh) == P(("i", ("j", "k")))


def test_tuple_leaf_minus1_0():
    """Test that P((-1, 0)) in a 3D mesh expands correctly."""
    mesh = Mesh(np.empty((2, 2, 2), dtype=object), ("i", "j", "k"))
    assert P((-1, 0))._expand_with_mesh(mesh) == P((("j", "k"), "i"))


def test_nested_tuple_specs():
    """Test P((0, -1), (1, 2)) with a 4D mesh."""
    mesh = Mesh(np.empty((2, 2, 2, 2), dtype=object), ("i", "j", "k", "l"))
    # First partition: (0, -1) mentions 0, remaining is 1,2,3 -> ("i", ("j", "k", "l"))
    # Second partition: (1, 2) mentions 1,2, no remaining -> ("j", "k")
    result = P((0, -1), (1, 2))._expand_with_mesh(mesh)
    assert result == P(("i", ("j", "k", "l")), ("j", "k"))


# ============================================================================
# EMPTY EXPANSION HANDLING
# ============================================================================

def test_minus1_all_axes_mentioned_1d():
    """Test P((0, -1)) in a 1D mesh where all axes are mentioned."""
    mesh = Mesh(np.empty((4,), dtype=object), ("i",))
    # 0 mentions i, -1 expands to nothing -> becomes None
    assert P((0, -1))._expand_with_mesh(mesh) == P("i")


def test_minus1_all_axes_mentioned_2d():
    """Test P((0, 1, -1)) in a 2D mesh where all axes are mentioned."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    # 0,1 mention all, -1 expands to nothing
    assert P((0, 1, -1))._expand_with_mesh(mesh) == P(("i", "j"))


def test_p0_minus1_all_mentioned():
    """Test P(0, 1, -1) when all axes are already mentioned."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    result = P(0, 1, -1)._expand_with_mesh(mesh)
    # 0 and 1 mention both axes, -1 is empty
    assert result == P("i", "j", None)


# ============================================================================
# ERROR CASES
# ============================================================================

def test_error_out_of_range_positive_index():
    """Test that out-of-range positive indices raise ValueError."""
    mesh = Mesh(np.empty((4,), dtype=object), ("i",))
    with pytest.raises(ValueError, match="out of range"):
        P(0, 1)._expand_with_mesh(mesh)


def test_error_out_of_range_negative_index():
    """Test that out-of-range negative indices raise ValueError."""
    mesh = Mesh(np.empty((4,), dtype=object), ("i",))
    with pytest.raises(ValueError, match="out of range"):
        P(-3)._expand_with_mesh(mesh)


def test_error_multiple_minus1_toplevel():
    """Test that multiple -1s at top level raise ValueError."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    with pytest.raises(ValueError, match="at most one"):
        P(-1, -1)._expand_with_mesh(mesh)


def test_error_multiple_minus1_in_nested():
    """Test that multiple -1s within a tuple are caught."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    with pytest.raises(ValueError, match="at most one"):
        P((-1, -1))._expand_with_mesh(mesh)


def test_error_minus1_in_different_tuples():
    """Test that -1 in different tuple leaves is caught."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    with pytest.raises(ValueError, match="at most one"):
        P((-1,), (-1,))._expand_with_mesh(mesh)


# ============================================================================
# COMPLEX AND INTEGRATION-LIKE TESTS
# ============================================================================

def test_complex_spec_4d_mesh():
    """Test a complex spec with a 4D mesh."""
    mesh = Mesh(np.empty((2, 2, 2, 2), dtype=object), ("i", "j", "k", "l"))
    # P(-1, (1, 2)) should:
    # - First partition: -1 expands to all unmentioned -> (i, j, k, l)
    # - Second partition: (1, 2) mentions j, k -> (j, k)
    result = P(-1, (1, 2))._expand_with_mesh(mesh)
    assert result == P(("i", "j", "k", "l"), ("j", "k"))


def test_all_none_then_wildcard():
    """Test P(None, None, -1) in a 3D mesh."""
    mesh = Mesh(np.empty((2, 2, 2), dtype=object), ("i", "j", "k"))
    # No axes mentioned in None entries, -1 expands to all
    result = P(None, None, -1)._expand_with_mesh(mesh)
    assert result == P(None, None, ("i", "j", "k"))


def test_interleaved_indices_and_wildcard():
    """Test P(0, None, -1) in a 3D mesh."""
    mesh = Mesh(np.empty((2, 2, 2), dtype=object), ("i", "j", "k"))
    # 0 mentions i, -1 expands to j, k
    result = P(0, None, -1)._expand_with_mesh(mesh)
    assert result == P("i", None, ("j", "k"))


def test_wildcard_at_different_positions():
    """Test -1 at different positions in longer specs."""
    mesh = Mesh(np.empty((2, 2, 2, 2), dtype=object), ("i", "j", "k", "l"))
    
    # -1 at end
    result1 = P(0, -1)._expand_with_mesh(mesh)
    assert result1 == P("i", ("j", "k", "l"))
    
    # -1 in middle
    result2 = P(0, -1, 3)._expand_with_mesh(mesh)
    assert result2 == P("i", ("j", "k"), "l")
    
    # -1 at start
    result3 = P(-1, 3)._expand_with_mesh(mesh)
    assert result3 == P(("i", "j", "k"), "l")


# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================

def test_named_specs_unchanged():
    """Test that named PartitionSpecs are unchanged."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    named_spec = P("i", "j")
    # Named specs should not have numeric indices, so no expansion needed
    assert named_spec._expand_with_mesh(mesh) == named_spec


def test_mixed_named_and_numeric():
    """Test that mixing named and numeric specs works."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    # P(0, "j") should expand to P("i", "j")
    result = P(0, "j")._expand_with_mesh(mesh)
    assert result == P("i", "j")


def test_none_specs_unchanged():
    """Test that None entries remain None."""
    mesh = Mesh(np.empty((2, 2), dtype=object), ("i", "j"))
    result = P(None, None)._expand_with_mesh(mesh)
    assert result == P(None, None)


# ============================================================================
# DIMENSION MATCHING VALIDATION TESTS
# ============================================================================

def test_spec_dimension_must_match():
    """Test that trying to expand a redundant spec doesn't break."""
    mesh = Mesh(np.empty((4,), dtype=object), ("i",))
    # This should work fine - spec dimensions are about the spec itself,
    # not about input arrays
    result = P(0, None)._expand_with_mesh(mesh)
    assert result == P("i", None)
