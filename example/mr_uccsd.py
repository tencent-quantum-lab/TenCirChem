import numpy as np
from pyscf.mcscf import CASSCF

from tencirchem import UCCSD
from tencirchem.molecule import water
from tencirchem.static.ci_utils import get_ci_strings

"""
Multi-reference UCCSD using CASSCF molecular obitals and ci coefficients
"""

as_elec = 2
as_orbital = 2

m = water(bond_length=3)
hf = m.HF()
hf.kernel()
casscf = CASSCF(hf, as_orbital, as_elec)
casscf.kernel()


# naive UCCSD, poor accuracy
uccsd = UCCSD(m)
uccsd.kernel()
uccsd.print_summary()
print(uccsd.energy(np.zeros_like(uccsd.params)))


# only using optimized mo coeff, also poor accuracy
uccsd = UCCSD(m, init_method="zeros", mo_coeff=casscf.mo_coeff, pick_ex2=False)
uccsd.kernel()
uccsd.print_summary()
print(uccsd.energy(np.zeros_like(uccsd.params)))


# optimized mo coeff + multi-reference initial state, good accuracy
uccsd = UCCSD(m, init_method="zeros", mo_coeff=casscf.mo_coeff, pick_ex2=False)
# set up the initial state. Embed CAS to the whole system
uccsd.init_state = np.zeros(uccsd.civector_size)
_, strs2addr = uccsd.get_ci_strings(strs2addr=True)
cas_configurations = [f"{bin(i)[2:]:0>4}" for i in get_ci_strings(2 * as_orbital, as_elec, False)]
for i, coeff in enumerate(casscf.ci.ravel()):
    c = cas_configurations[i]
    template = "0" * (uccsd.nv - as_orbital + as_elec // 2) + "{}" + "1" * (uccsd.no - as_elec // 2)
    c_alpha = template.format(c[:as_orbital])
    c_beta = template.format(c[as_orbital:])
    addr = uccsd.get_addr(c_alpha + c_beta)
    uccsd.init_state[addr] = coeff
uccsd.kernel()
uccsd.print_summary()
uccsd.civector(np.zeros_like(uccsd.params)), uccsd.civector(uccsd.params)
print(uccsd.energy(np.zeros_like(uccsd.params)))
