import numpy as np
from tensorcircuit import Circuit

from tencirchem import UCC, HEA, set_backend
from tencirchem.molecule import h4

K = set_backend("jax")

ucc = UCC(h4)
# reference energies
ucc.print_energy()


# symmetry preserving ansatz: npjqi 6:10, 2020
n_layers = 5
n_params = n_layers * (ucc.n_qubits - 2)
def circuit(params):
    params = params.reshape(n_layers, ucc.n_qubits - 2)
    c = Circuit(ucc.n_qubits)
    # HF initial state
    for i in range(ucc.n_elec // 2):
        c.x(ucc.n_qubits - 1 - i)
        c.x(ucc.n_qubits // 2 - 1 - i)
    for l in range(n_layers):
        indices = list(range(0, ucc.n_qubits - 1, 2)) + list(range(1, ucc.n_qubits - 1, 2))
        for i in indices:
            if i < ucc.n_qubits // 2 - 1:
                theta = params[l, i]
            elif i == ucc.n_qubits // 2 - 1:
                # preserve s_z
                theta = 0
            else:
                theta = params[l, i-1]
            unitary = K.convert_to_tensor([[1, 0, 0, 0],
                       [0, K.cos(theta), K.sin(theta), 0],
                       [0, K.sin(theta), -K.cos(theta), 0],
                       [0, 0, 0, 1]])
            c.any(i, (i + 1), unitary=unitary)
    return c


es = []
for i in range(10):
    init_guess = (np.random.rand(n_params) - 0.5)
    hea = HEA(ucc.h_qubit_op, circuit, init_guess, engine="tensornetwork")
    # default parameter shift doesn't work for this type of circuit
    hea.grad = "autodiff"
    e = hea.kernel()
    es.append(e)
print(sorted(es))
