def test_custom_jvp_repr_is_function_like():
    import jax
    import jax.nn as nn

    r = repr(nn.relu)

    # Should look like a function
    assert r.startswith("<function")

    # Should contain the public function name
    assert "relu" in r

    # Should not expose internal wrapper details
    assert "custom_jvp" not in r
