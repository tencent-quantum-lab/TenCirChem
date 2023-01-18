from tencirchem import set_backend
from tencirchem import UCCSD
from tencirchem.molecule import h_chain


m = h_chain(n_h=6)
uccsd = UCCSD(m, engine="civector")

# switch backend at runtime
for backend_str in ["numpy", "jax"]:
    set_backend(backend_str)
    print(uccsd.kernel())
    uccsd.print_summary()
