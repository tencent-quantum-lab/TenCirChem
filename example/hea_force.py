#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

from pyscf.mcscf import CASCI

from tencirchem import HEA
from tencirchem.molecule import n2

# normal PySCF workflow
hf = n2(2).HF()
print(hf.kernel())
casci = CASCI(hf, 2, 2)
# set the FCI solver for CASSCF to be HEA
casci.canonicalization = False  # prevent changing mo_coeffs
casci.fcisolver = HEA.as_pyscf_solver(n_layers=1)
print(casci.kernel()[0])
nuc_grad = casci.nuc_grad_method().kernel()
print(nuc_grad)
