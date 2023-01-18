from pyscf import M

from tencirchem.static.hea import HEA
from tencirchem import UCCSD, set_backend

set_backend("jax")

d = 1
mol = M(atom=[["H", 0, 0, 0], ["Li", d, 0, 0]], charge=0, symmetry=True)
# reference energy
ucc = UCCSD(mol)
ucc.print_energy()

# move active orbitals to the middle
hf = mol.HF()
hf.kernel()
hf.mo_coeff = hf.mo_coeff[:, [0, 1, 2, 5, 3, 4]]

# reference energy with AS
# Li 1s, 2py, 2pz frozen
ucc = UCCSD(mol, active_space=(2, 3), mo_coeff=hf.mo_coeff)
ucc.print_energy()

# HEA run
hea = HEA.ry(ucc.int1e, ucc.int2e, ucc.n_elec, ucc.e_core, 3, engine="tensornetwork")
hea.grad = "autodiff"

hea.kernel()
hea.print_summary()
