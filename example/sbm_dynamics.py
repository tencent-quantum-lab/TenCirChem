from tencirchem import Op, TimeEvolution, set_backend, set_dtype
from tencirchem.dynamic.model import sbm

set_backend("jax")
set_dtype("complex128")

epsilon = delta = 1
nmode = 1
omega_list = [1]
g_list = [1]

nbas = 6
n_layers = 3

ham_terms = sbm.get_ham_terms(epsilon, delta, nmode, omega_list, g_list)
basis = sbm.get_basis(omega_list, nlevels=nbas)

te = TimeEvolution(ham_terms, basis, "gray", {}, n_layers, eps=1e-4, property_op_dict={"Z": Op("Z", "spin")})

tau = 0.1
steps = 100

for n in range(steps):
    te.kernel(tau)

print(te.property_dict)
