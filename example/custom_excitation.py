import numpy as np

from tencirchem import UCC, PUCCD
from tencirchem.molecule import h4


puccd = PUCCD(h4)
puccd.kernel()


ucc = UCC(h4)
# customize UCC excitation operator to mimic pUCCD
# See the document for the convention of the excitation operators
ucc.ex_ops = [
    (7, 3, 0, 4),
    (6, 2, 0, 4),
    (7, 3, 1, 5),
    (6, 2, 1, 5),
]
ucc.param_ids = puccd.param_ids
ucc.init_guess = puccd.params
ucc.kernel()

np.testing.assert_allclose(ucc.e_ucc, puccd.e_puccd, atol=1e-5)
