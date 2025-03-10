#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

"""
Variational basis state encoder for the ground state of the Holstein model.
2 qubit or each phonon mode.
https://arxiv.org/abs/2301.01442
"""

import numpy as np
import scipy
from opt_einsum import contract
import tensorcircuit as tc

from tencirchem import set_backend, Op, BasisSHO, BasisSimpleElectron, Mpo, Model
from tencirchem.dynamic import get_ansatz, qubit_encode_op_grouped, qubit_encode_basis
from tencirchem.utils import scipy_opt_wrap
from tencirchem.applications.vbe_lib import get_psi_indices, get_contracted_mpo, get_contract_args

backend = set_backend("jax")

nsite = 3
omega = 1
v = 1
# two qubit for each mode
n_qubit_per_mode = 2
nbas_v = 1 << n_qubit_per_mode

# -1 for electron dof, natural numbers for phonon dof
dof_nature = np.array([-1, 0, 0, -1, 1, 1, -1, 2, 2])
# physical index for phonon mode
b_dof_pidx = np.array([1, 3, 5])

psi_idx_top, psi_idx_bottom, b_dof_vidx = get_psi_indices(dof_nature, b_dof_pidx, n_qubit_per_mode)

n_dof = len(dof_nature)
psi_shape2 = [2] * n_dof

c = tc.Circuit(nsite * 3)
c.X(0)
n_layers = 3


def get_vha_terms():
    # variational Hamiltonian ansatz (vha) terms

    g = 1  # dummy value, doesn't matter
    ansatz_terms = []
    for i in range(nsite):
        j = (i + 1) % nsite
        ansatz_terms.append(Op(r"a^\dagger a", [i, j], v) - Op(r"a^\dagger a", [j, i], v))
        ansatz_terms.append(Op(r"a^\dagger a b^\dagger-b", [i, i, (i, 0)], g * omega))

    basis = []
    for i in range(nsite):
        basis.append(BasisSimpleElectron(i))
        basis.append(BasisSHO((i, 0), omega, nbas_v))

    ansatz_terms, _ = qubit_encode_op_grouped(ansatz_terms, basis, boson_encoding="gray")

    # More flexible ansatz by decoupling the parameters in the e-ph coupling term
    ansatz_terms_extended = []
    for i in range(nsite):
        ansatz_terms_extended.extend([ansatz_terms[2 * i]] + ansatz_terms[2 * i + 1])
    spin_basis = qubit_encode_basis(basis, boson_encoding="gray")
    return ansatz_terms_extended, spin_basis


ansatz_terms, spin_basis = get_vha_terms()
ansatz = get_ansatz(ansatz_terms, spin_basis, n_layers, c)


def cost_fn(params, h):
    state = ansatz(params)
    return (state.conj() @ (h @ state)).squeeze().real


vg = backend.jit(backend.value_and_grad(cost_fn))
opt_fn = scipy_opt_wrap(vg)


def get_ham_terms_and_basis(g, nbas):
    terms = []
    for i in range(nsite):
        terms.append(Op(r"b^\dagger b", (i, 0), omega))
        terms.append(Op(r"a^\dagger a b^\dagger+b", [i, i, (i, 0)], g * omega))
        j = (i + 1) % nsite
        terms.append(Op(r"a^\dagger a", [i, j], -v))
        terms.append(Op(r"a^\dagger a", [j, i], -v))

    basis = []
    for i in range(nsite):
        basis.append(BasisSimpleElectron(i))
        basis.append(BasisSHO((i, 0), omega, nbas))

    return terms, basis


def solve_b_array(psi, h_mpo, b_array, i):
    nbas = b_array.shape[-1]
    args = get_contract_args(psi, h_mpo, b_array, i, n_qubit_per_mode, psi_idx_top, psi_idx_bottom, b_dof_pidx)
    k = b_dof_pidx[i]
    # output indices
    args.append(
        [
            f"v-{k}-0-bottom",
            f"v-{k}-1-bottom",
            f"p-{k}-bottom",
            f"v-{k}-0-top",
            f"v-{k}-1-top",
            f"p-{k}-top",
            "mpo-0",
            f"mpo-{len(h_mpo)}",
        ]
    )
    contracted_h = contract(*args).reshape(4, nbas, 4, nbas)
    nroot = 3

    def f(x):
        x = x.reshape(nroot, 4, nbas)
        p = contract("abc, abd -> acd", x.conj(), x)
        return contract("abcd, kab, kde -> kce", contracted_h, x, (np.array([np.eye(nbas)] * nroot) - p)).ravel()

    sols = scipy.optimize.root(f, [b_array[i].flatten()] * nroot, method="df-sane").x.reshape(3, 4, nbas)

    sols = list(sols) + [b_array[i].copy()]
    b_array = b_array.copy()
    es = []
    for k, new_b in enumerate(sols):
        if not np.allclose(new_b @ new_b.T, np.eye(4)):
            # print(f"Enforcing orthogonality for the {k}th root of b_array[{i}]")
            new_b = np.linalg.qr(new_b.T)[0].T
        b_array[i] = new_b
        e = psi @ get_contracted_mpo(h_mpo, b_array, n_qubit_per_mode, b_dof_pidx, psi_idx_top + psi_idx_bottom) @ psi
        es.append(e)
    # print(np.array(es))
    lowest_id = np.argmin(es)
    return sols[lowest_id]


def main():
    vqe_e = []
    thetas = np.zeros(len(ansatz_terms) * n_layers)

    for g in [1.5, 3]:
        for nbas in [4, 8, 12, 16, 20, 24, 28, 32]:
            print(f"g: {g}, nbas: {nbas}")

            b_list = []
            for i in range(max(dof_nature) + 1):
                b = np.eye(nbas)[:nbas_v]  # nbas_dummy * nbas
                b_list.append(b)
            b_array = np.array(b_list)

            terms, basis = get_ham_terms_and_basis(g, nbas)
            model = Model(basis, terms)
            h_mpo = Mpo(model)

            for i_iter in range(10):
                h_contracted = get_contracted_mpo(
                    h_mpo, b_array, n_qubit_per_mode, b_dof_pidx, psi_idx_top + psi_idx_bottom
                )
                opt_res = scipy.optimize.minimize(
                    opt_fn, args=(h_contracted,), x0=thetas / 2, jac=True, method="L-BFGS-B"
                )
                print(f"Iter {i_iter} VQE energy: {opt_res.fun}")
                thetas = opt_res.x
                psi = np.array(ansatz(thetas).real)
                for i in range(len(b_array)):
                    b_array[i] = solve_b_array(psi, h_mpo, b_array, i)
            vqe_e.append(opt_res.fun)

        print(vqe_e)


if __name__ == "__main__":
    main()
