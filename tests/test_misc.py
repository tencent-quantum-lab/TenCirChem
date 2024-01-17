import numpy as np
import pytest

from tencirchem import clear_cache, UCCSD
from tencirchem.molecule import h4, ch4, benzene


def test_clear_cache():
    uccsd = UCCSD(h4)
    e1 = uccsd.kernel()
    clear_cache()
    e2 = uccsd.kernel()
    np.testing.assert_allclose(e2, e1)


@pytest.mark.parametrize("mol", [ch4, benzene])
def test_molecule(mol):
    mol = mol()
    hf = mol.HF()
    hf.kernel()
