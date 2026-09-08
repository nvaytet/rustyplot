import numpy as np
import pytest

from rustyplot._data import to_colors, to_float32, to_sizes


def test_to_float32_converts_dtype_and_keeps_values():
    out = to_float32(np.array([1, 2, 3], dtype=np.int64), "x")
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_to_float32_accepts_plain_lists():
    np.testing.assert_array_equal(to_float32([1.0, 2.0], "x"), [1.0, 2.0])


def test_to_float32_rejects_2d():
    with pytest.raises(ValueError, match="one-dimensional"):
        to_float32(np.zeros((2, 2)), "x")


def test_to_float32_rejects_empty():
    with pytest.raises(ValueError, match="nothing to plot"):
        to_float32([], "x")


def test_scalar_size_is_broadcast():
    np.testing.assert_array_equal(to_sizes(5.0, 3), [5.0, 5.0, 5.0])


def test_size_length_is_validated():
    with pytest.raises(ValueError, match="has 2 elements"):
        to_sizes([1.0, 2.0], 3)


def test_default_color_is_repeated_per_point():
    out = to_colors(None, 3)
    assert out.shape == (12,)
    np.testing.assert_allclose(out[:4], out[4:8])


def test_hex_color_is_parsed():
    np.testing.assert_allclose(to_colors("#ff0000", 1), [1.0, 0.0, 0.0, 1.0])


def test_hex_color_with_alpha():
    np.testing.assert_allclose(
        to_colors("#00ff0080", 1), [0.0, 1.0, 0.0, 128 / 255], atol=1e-6
    )


def test_rgb_array_gets_opaque_alpha():
    out = to_colors(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 2)
    np.testing.assert_allclose(out, [1, 0, 0, 1, 0, 0, 1, 1])


def test_color_shape_is_validated():
    with pytest.raises(ValueError, match="must be a single RGB"):
        to_colors(np.zeros((5, 4)), 2)


@pytest.mark.parametrize("module", ["pandas", "xarray", "scipp"])
def test_planned_libraries_get_a_helpful_error(module):
    pytest.importorskip(module)
    lib = __import__(module)
    if module == "pandas":
        data = lib.Series([1.0, 2.0])
    elif module == "xarray":
        data = lib.DataArray(np.array([1.0, 2.0]))
    else:
        data = lib.array(dims=["x"], values=np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match="does not accept"):
        to_float32(data, "x")
