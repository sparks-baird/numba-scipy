import itertools

import pytest

import numpy as np
from numpy.testing import assert_allclose
import numba
import scipy.special as sc
import numba_scipy.special
from numba_scipy.special import signatures as special_signatures

NUMBA_TYPES_TO_TEST_POINTS = {
    numba.types.float64: np.array(
        [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0],
        dtype=np.float64
    ),
    numba.types.float32: np.array(
        [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0],
        dtype=np.float32
    ),
    numba.types.long_: np.array(
        [-100, -10, -1, 0, 1, 10, 100],
        dtype=np.int_
    )
}

SKIP_LIST = {
    # Should be fixed by https://github.com/scipy/scipy/pull/10455
    (
        'hyperu',
        (numba.types.float64,) * 3
    ),
    # Sometimes returns nan, sometimes returns inf. Likely a SciPy bug.
    (
        'eval_jacobi',
        (numba.types.float64,) * 4
    ),
    # Sometimes returns nan, sometimes returns inf. Likely a SciPy bug.
    (
        'eval_sh_jacobi',
        (numba.types.float64,) * 4
    )
}


def get_parametrize_arguments():
    signatures = special_signatures.name_to_numba_signatures.items()
    for name, specializations in signatures:
        for signature in specializations:
            yield name, signature


@pytest.mark.parametrize(
    'name, specialization',
    get_parametrize_arguments(),
)
def test_function(name, specialization):
    if (name, specialization) in SKIP_LIST:
        pytest.xfail()

    f = getattr(sc, name)

    @numba.njit
    def wrapper(*args):
        return f(*args)

    args = itertools.product(*(
        NUMBA_TYPES_TO_TEST_POINTS[numba_type] for numba_type in specialization
    ))
    for arg in args:
        overload_value = wrapper(*arg)
        scipy_value = f(*arg)
        if np.isnan(overload_value):
            assert np.isnan(scipy_value)
        else:
            rtol = 2**8 * np.finfo(scipy_value.dtype).eps
            assert_allclose(overload_value, scipy_value, atol=0, rtol=rtol)
