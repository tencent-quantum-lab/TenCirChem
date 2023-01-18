import numpy as np

from tencirchem import clear_cache, UCCSD
from tencirchem.molecule import h4


def test_clear_cache():
    uccsd = UCCSD(h4)
    e1 = uccsd.kernel()
    clear_cache()
    e2 = uccsd.kernel()
    np.testing.assert_allclose(e2, e1)
