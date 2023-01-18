from tencirchem import UCCSD, HEA
from tencirchem.molecule import h2

# use the `UCCSD` class to access the integrals
uccsd = UCCSD(h2)

# use the `HEA` class for noisy circuits. A 1-layer Ry ansatz is used
hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, n_layers=1, engine="tensornetwork-noise")
hea.kernel()
hea.print_summary()

# noiseless energy
print(hea.energy(engine="tensornetwork"))
# the energy with gate noise and measurement noise
print(hea.energy(engine="tensornetwork-noise&shot"))
