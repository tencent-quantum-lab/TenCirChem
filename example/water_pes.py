import numpy as np

# TenCirChem accepts PySCF Mole as input
from pyscf import M


# assuming the same O-H length for the two bonds
def get_h2o_m(bond_angle, bond_length):
    phi = bond_angle / 2
    r = bond_length
    O = ["O", 0, 0, 0]
    H1 = ["H", -r * np.sin(phi), r * np.cos(phi), 0]
    H2 = ["H", r * np.sin(phi), r * np.cos(phi), 0]
    return M(atom=[O, H1, H2], basis="631G(d)")


# PES range
bond_angles = np.array([104.45]) / 180 * np.pi
# in angstrom
bond_lengths = np.linspace(0.6, 2, 29)

# build molecules
from itertools import product

moles = []
for bond_angle, bond_length in product(bond_angles, bond_lengths):
    moles.append(get_h2o_m(bond_angle, bond_length))


from tencirchem import UCCSD, set_dtype, set_backend

set_dtype("complex128")
set_backend("cupy")
e_hf_list = []
e_mp2_list = []
e_ccsd_list = []
e_ucc_list = []
e_fci_list = []
for m in moles:
    uccsd = UCCSD(m, run_ccsd=True, run_fci=True, active_space=(8, 17))
    uccsd.kernel()
    print(m.atom)
    uccsd.print_summary()
    e_hf_list.append(uccsd.e_hf)
    e_mp2_list.append(uccsd.e_mp2)
    e_ccsd_list.append(uccsd.e_ccsd)
    e_ucc_list.append(uccsd.e_ucc)
    e_fci_list.append(uccsd.e_fci)

print(e_hf_list)
print(e_mp2_list)
print(e_ccsd_list)
print(e_ucc_list)
print(e_fci_list)
