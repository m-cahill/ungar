import numpy as np
import pytest
from ungar.tensor import CardTensor, CardTensorSpec


def test_empty_shape() -> None:
    tensor = CardTensor.empty(["a", "b"])
    assert tensor.data.shape == (4, 14, 2)
    assert tensor.data.dtype == bool
    assert tensor.spec.num_planes == 2
    assert tensor.spec.plane_names == ("a", "b")


def test_immutability() -> None:
    tensor = CardTensor.empty(["a"])
    assert tensor.data.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        tensor.data[0, 0, 0] = True


def test_invalid_shape() -> None:
    spec = CardTensorSpec(("a",))
    data = np.zeros((4, 14, 2), dtype=bool)  # Wrong depth
    with pytest.raises(ValueError, match="CardTensor data shape"):
        CardTensor(data, spec)


def test_invalid_dtype() -> None:
    spec = CardTensorSpec(("a",))
    data = np.zeros((4, 14, 1), dtype=int)  # Wrong dtype
    with pytest.raises(TypeError, match="dtype must be bool"):
        CardTensor(data, spec)


def test_invalid_plane_name() -> None:
    with pytest.raises(ValueError, match="must be snake_case"):
        CardTensorSpec(("InvalidName",))

    with pytest.raises(ValueError, match="must be snake_case"):
        CardTensorSpec(("invalid-name",))

    # Valid names
    CardTensorSpec(("valid_name", "also_valid_123", "x"))
