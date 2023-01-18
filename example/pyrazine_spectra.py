import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from renormalizer.mps import Mpo
from renormalizer.model import Model
from renormalizer.utils.constant import fs2au
import tensorcircuit as tc

from tencirchem import set_backend, set_dtype
from tencirchem.dynamic.model import pyrazine
from tencirchem.dynamic.transform import qubit_encode_op, qubit_encode_basis
from tencirchem.dynamic.time_derivative import get_ansatz, get_jacobian_func, get_deriv

set_backend("jax")
set_dtype("complex128")

ham_terms = pyrazine.get_ham_terms()
basis = pyrazine.get_basis(2)
h1 = Mpo(Model(basis, ham_terms)).todense()

spin_ham_terms, constant = qubit_encode_op(ham_terms, basis)
spin_basis = qubit_encode_basis(basis)
h2 = Mpo(Model(spin_basis, spin_ham_terms)).todense()
h2 += np.eye(len(h2)) * constant

np.testing.assert_allclose(h2, h1)


l = 2
tau = 0.6 * fs2au
steps = 100

circuit = tc.Circuit(len(spin_basis))
circuit.X(0)
psi0 = circuit.state()

theta = np.zeros(l * len(spin_ham_terms), dtype=np.float64)
ansatz = get_ansatz(spin_ham_terms, spin_basis, l, circuit)
jacobian_func = get_jacobian_func(ansatz)


def scipy_deriv(t, _theta):
    return get_deriv(ansatz, jacobian_func, _theta, h2)


autocorrelation_time = [0]
autocorrelation = [1]
theta_list = []
for n in range(steps):
    time0 = time.time()
    sol = solve_ivp(scipy_deriv, [n * tau, (n + 1) * tau], theta)
    time1 = time.time()
    theta = sol.y[:, -1]
    theta_list.append(theta)
    psi = ansatz(theta)
    psi_conj = ansatz(-theta)
    autocorrelation_time.append(2 * n * tau)
    autocorrelation.append(psi_conj.conj() @ psi)
    psi_exact = scipy.linalg.expm(-1j * h1 * (n + 1) * tau) @ psi0.reshape(-1)
    psi_exact = tc.backend.reshape(psi_exact, [-1])
    print(
        f"step: {n}",
        f"time: {n * tau:.2f}",
        f"wall time: {time1 - time0:.2f}",
        f"substeps: {sol.y.shape[1]}",
        "exact:",
        tc.expectation([tc.gates.z(), [0]], ket=psi_exact).real,
        "variational:",
        tc.expectation([tc.gates.z(), [0]], ket=psi).real,
    )

autocorrelation_time = np.array(autocorrelation_time)
autocorrelation = np.array(autocorrelation)
