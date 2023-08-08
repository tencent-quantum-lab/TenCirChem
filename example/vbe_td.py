#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

"""
Variational basis state encoder for the dynamics of the spin-boson model.
2 qubit or each phonon mode.
https://arxiv.org/abs/2301.01442
"""

import numpy as np
from scipy.integrate import solve_ivp
from opt_einsum import contract
import tensorcircuit as tc

from tencirchem import set_backend, Op, Mpo, Model, OpSum
from tencirchem.dynamic import get_ansatz, get_deriv, get_jacobian_func, qubit_encode_basis, sbm
from tencirchem.applications.vbe_lib import get_psi_indices, get_contracted_mpo, get_contract_args

set_backend("jax")

epsilon = 0
delta = 1
omega_list = [0.5, 1]
g_list = [0.25, 1]

nmode = len(omega_list)
assert nmode == len(g_list)

# two qubit for each mode
n_qubit_per_mode = 2
nbas_v = 1 << n_qubit_per_mode

# -1 for electron dof, natural numbers for phonon dof
dof_nature = np.array([-1] + [0] * n_qubit_per_mode + [1] * n_qubit_per_mode)
b_dof_pidx = np.array([1, 2])

n_dof = len(dof_nature)
psi_shape2 = [2] * n_dof

psi_idx_top, psi_idx_bottom, b_dof_vidx = get_psi_indices(dof_nature, b_dof_pidx, n_qubit_per_mode)


def get_model(epsilon, delta, nmode, omega_list, g_list, nlevels):
    ham_terms = sbm.get_ham_terms(epsilon, delta, nmode, omega_list, g_list)
    basis = sbm.get_basis(omega_list, nlevels)
    return Model(basis, ham_terms)


nbas = 16

b_shape = tuple([2] * n_qubit_per_mode + [nbas])

assert len(omega_list) == nmode
assert len(g_list) == nmode
model = get_model(epsilon, delta, nmode, omega_list, g_list, [nbas] * nmode)

h_mpo = Mpo(model)

circuit = tc.Circuit(1 + nmode * n_qubit_per_mode)
psi0 = circuit.state()
n_layers = 3


def get_vha_terms():
    basis = sbm.get_basis(omega_list, [nbas_v] * nmode)
    spin_basis = qubit_encode_basis(basis, "gray")

    spin_ham_terms = OpSum([Op("X", ["spin"], 1.0)])
    for i in range(nmode):
        complete_list = []
        for j in range(n_qubit_per_mode):
            complete = OpSum()
            dof = (f"v{i}", f"TCCQUBIT-{j}")
            for symbol in "IXYZ":
                complete += Op(symbol, dof)
            complete_list.append(complete)
        complete_real = complete_list[0]
        for c in complete_list[1:]:
            complete_real = complete_real * c
        spin_ham_terms.extend(complete_real)
        spin_ham_terms.extend(Op("Z", "spin") * complete_real)
    spin_ham_terms = OpSum([op.squeeze_identity() for op in spin_ham_terms.simplify() if not op.is_identity]).simplify()
    return spin_ham_terms, spin_basis


spin_ham_terms, spin_basis = get_vha_terms()


theta0 = np.zeros(n_layers * len(spin_ham_terms), dtype=np.float64)
ansatz = get_ansatz(spin_ham_terms, spin_basis, n_layers, psi0)
jacobian_func = get_jacobian_func(ansatz)


def deriv_fun(t, theta_and_b):
    theta = theta_and_b[: len(theta0)]
    psi = ansatz(theta)
    b_array = theta_and_b[len(theta0) :].reshape(nmode, nbas_v, nbas)

    h_contracted = get_contracted_mpo(h_mpo, b_array, n_qubit_per_mode, b_dof_pidx, psi_idx_top + psi_idx_bottom)
    theta_deriv = get_deriv(ansatz, jacobian_func, theta, h_contracted)

    psi = psi.reshape(psi_shape2)
    b_deriv_list = []
    for i in range(nmode):
        b = b_array[i]
        # calculate rho
        indices_base = [("contract", ii) for ii in range(n_dof)]
        psi_top_indices = indices_base.copy()
        psi_bottom_indices = indices_base.copy()
        for j in b_dof_vidx[i]:
            psi_top_indices[j] = ("top", j)
            psi_bottom_indices[j] = ("bottom", j)
        out_indices = [("top", j) for j in b_dof_vidx[i]] + [("bottom", j) for j in b_dof_vidx[i]]
        args = [psi.conj(), psi_top_indices, psi, psi_bottom_indices, out_indices]
        rho = contract(*args).reshape(1 << n_qubit_per_mode, 1 << n_qubit_per_mode)
        # rho_inv = regularized_inversion(rho, 1e-6)
        from scipy.linalg import pinv

        rho += np.eye(len(rho)) * 1e-5
        rho_inv = pinv(rho)

        b = b.reshape(nbas_v, nbas)
        # projector
        proj = b.conj().T @ b

        # derivative
        args = get_contract_args(psi, h_mpo, b_array, i, n_qubit_per_mode, psi_idx_top, psi_idx_bottom, b_dof_pidx)
        k = b_dof_pidx[i]
        args.append(b_array[i].reshape(b_shape))
        args.append([f"v-{k}-{l}-bottom" for l in range(n_qubit_per_mode)] + [f"p-{k}-bottom"])
        # output indices
        args.append([f"v-{k}-{l}-top" for l in range(n_qubit_per_mode)] + [f"p-{k}-top", "mpo-0", f"mpo-{len(h_mpo)}"])

        # take transpose to be compatible with previous code
        b_deriv = contract(*args).squeeze().reshape(nbas_v, nbas).T
        b_deriv = np.einsum("bf, bg -> fg", b_deriv, np.eye(nbas) - proj)
        b_deriv = -1j * np.einsum("fg, fh -> hg", b_deriv, rho_inv.T)
        b_deriv_list.append(b_deriv)
    return np.concatenate([theta_deriv, np.array(b_deriv_list).ravel()])


def main():
    b_list = []
    for _ in range(nmode):
        b = np.eye(nbas)[:nbas_v]  # nbas_v * nbas
        b_list.append(b)
    theta_and_b = np.concatenate([theta0, np.array(b_list).ravel()]).astype(complex)
    z_list = [1]
    x_list = [0]

    tau = 0.1
    steps = 100

    dummy_model = get_model(epsilon, delta, nmode, omega_list, g_list, [nbas_v] * nmode)
    z_op = Mpo(dummy_model, Op("Z", "spin", factor=1)).todense()
    x_op = Mpo(dummy_model, Op("X", "spin", factor=1)).todense()

    for n in range(steps):
        print(n)
        sol = solve_ivp(deriv_fun, [n * tau, (n + 1) * tau], theta_and_b)
        theta_and_b = sol.y[:, -1]
        theta = theta_and_b[: len(theta0)]
        psi = ansatz(theta)
        z = psi.conj().T @ (z_op @ psi)
        x = psi.conj().T @ (x_op @ psi)
        z_list.append(z.real)
        x_list.append(x.real)

    print(z_list)


if __name__ == "__main__":
    main()
