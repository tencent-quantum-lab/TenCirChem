import numpy as np
from pyscf import M
from pyscf.fci import FCI
from tencirchem import UCCSD, set_dtype

set_dtype("complex128")

# TenCirChem uses PySCF `Mole` to represent molecules
m = M(atom=[["H", 0, 0, 0], ["H", 0, 0, 1]])
uccsd = UCCSD(m)
uccsd.kernel()
uccsd_rdm2 = uccsd.make_rdm2(basis="MO")


hf = m.HF()
hf.kernel()
fci = FCI(hf)
# TenCirChem civector/rdm follows the convention of PySCF
fci_rdm2 = fci.make_rdm2(uccsd.civector(), 2, 2)

np.testing.assert_allclose(uccsd_rdm2, fci_rdm2, atol=1e-7)
