import numpy as np
import importlib

# Importing the module should ensure np.float is available
import musicgen_stems_continue2  # noqa: F401

def test_numpy_float_alias_defined():
    assert hasattr(np, "float")
    assert np.float is float
