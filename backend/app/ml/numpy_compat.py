from __future__ import annotations

import sys

import numpy as np
from numpy.random import _pickle as numpy_random_pickle


def install_numpy_pickle_compat() -> None:
    """Alias NumPy 2.x private module paths for older runtimes during unpickling."""
    module_aliases = {
        "numpy._core": np.core,
        "numpy._core.numeric": np.core.numeric,
        "numpy._core.multiarray": np.core.multiarray,
        "numpy._core.umath": np.core.umath,
        "numpy._core._multiarray_umath": np.core._multiarray_umath,
    }
    for alias, module in module_aliases.items():
        sys.modules.setdefault(alias, module)

    original_ctor = numpy_random_pickle.__bit_generator_ctor
    if getattr(original_ctor, "_foresight_compat", False):
        return

    def _compat_bit_generator_ctor(bit_generator_name="MT19937"):
        if not isinstance(bit_generator_name, str):
            bit_generator_name = getattr(bit_generator_name, "__name__", str(bit_generator_name))
        return original_ctor(bit_generator_name)

    _compat_bit_generator_ctor._foresight_compat = True  # type: ignore[attr-defined]
    numpy_random_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
