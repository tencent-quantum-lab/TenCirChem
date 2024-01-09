#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

from pyscf.mcscf import CASCI
from pyscf.geomopt.berny_solver import optimize

# need to install pyberny before running the manuscript.

from tencirchem import HEA
from tencirchem.molecule import h2o

# normal PySCF workflow
hf = h2o(1.5).HF()
print(hf.kernel())
casci = CASCI(hf, 2, 2)
# set the FCI solver for CASSCF to be HEA
casci.canonicalization = False  # prevent changing mo_coeffs
casci.fcisolver = HEA.as_pyscf_solver(n_layers=1)
print(casci.kernel()[0])
conv_params = {
    "gradientmax": 1e-3,
    "gradientrms": 1e-4,
    "stepmax": 1e-3,
    "steprms": 1e-4,
}
mol_eq = optimize(casci, **conv_params)
