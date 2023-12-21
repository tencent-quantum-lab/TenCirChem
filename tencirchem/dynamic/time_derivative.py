#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import logging
from typing import List, Union

import numpy as np
import scipy
from renormalizer.model.basis import BasisSet
from renormalizer import Op
import tensorcircuit as tc

from tencirchem.utils.backend import jit
from tencirchem.utils.circuit import evolve_pauli


logger = logging.getLogger(__name__)


def construct_ansatz_op(ham_terms: List[Op], spin_basis: List[BasisSet]):
    dof_idx_dict = {b.dof: i for i, b in enumerate(spin_basis)}
    ansatz_op_list = []

    for i, op in enumerate(ham_terms):
        logger.info(f"Ansatz operator: {i}, {op}")

        op_mat = 1
        name = ""
        for symbol in op.split_symbol:
            if symbol in ["X", "x", "sigma_x"]:
                op_mat = np.kron(op_mat, tc.gates._x_matrix)
                name += "X"
            elif symbol in ["Y", "y", "sigma_y"]:
                op_mat = np.kron(op_mat, tc.gates._y_matrix)
                name += "Y"
            else:
                if symbol not in ["Z", "z", "sigma_z"]:
                    raise ValueError(f"Hamiltonian must be sum of Pauli strings, got term {op}")
                op_mat = np.kron(op_mat, tc.gates._z_matrix)
                name += "Z"
        qubit_idx_list = [dof_idx_dict[dof] for dof in op.dofs]
        ansatz_op_list.append((op_mat, op.factor, name, qubit_idx_list))

    return ansatz_op_list


def get_circuit(ansatz_terms, spin_basis, n_layers, init_state, params, param_ids=None, compile_evolution=False):
    if param_ids is None:
        param_ids = list(range(len(ansatz_terms)))

    params = tc.backend.reshape(params, [n_layers, max(param_ids) + 1])

    ansatz_terms_grouped = []
    for term in ansatz_terms:
        if isinstance(term, Op):
            ansatz_terms_grouped.append([term])
        else:
            ansatz_terms_grouped.append(term)
    assert isinstance(ansatz_terms_grouped[0], list) and isinstance(ansatz_terms_grouped[0][0], Op)

    ansatz_op_list_grouped = []
    for ansatz_terms in ansatz_terms_grouped:
        ansatz_op_list = construct_ansatz_op(ansatz_terms, spin_basis)
        factors = np.array([factor for _, factor, _, _ in ansatz_op_list])
        assert np.allclose(factors.real, 0) or np.allclose(factors.imag, 0)
        ansatz_op_list_grouped.append(ansatz_op_list)

    if isinstance(init_state, tc.Circuit):
        c = tc.Circuit.from_qir(init_state.to_qir(), circuit_params=init_state.circuit_param)
    else:
        c = tc.Circuit(len(spin_basis), inputs=init_state)

    for i in range(0, n_layers):
        for j, ansatz_op_list in enumerate(ansatz_op_list_grouped):
            param_id = param_ids[j]
            theta = params[i, param_id]
            for ansatz_op, coeff, name, qubit_idx_list in ansatz_op_list:
                if coeff.real == 0:
                    coeff = coeff.imag
                if not compile_evolution:
                    np.testing.assert_allclose(ansatz_op.conj().T @ ansatz_op, np.eye(len(ansatz_op)))
                    name = f"exp(-iθ{name})"
                    c.exp1(*qubit_idx_list, unitary=ansatz_op, theta=coeff * theta, name=name)
                else:
                    pauli_string = tuple(zip(qubit_idx_list, name))
                    c = evolve_pauli(c, pauli_string, theta=2 * coeff * theta)
    return c


def one_trotter_step(ham_terms, spin_basis, init_state, dt, inplace=False):
    """
    one step first order trotter decompostion
    """
    ansatz_op_list = construct_ansatz_op(ham_terms, spin_basis)

    if isinstance(init_state, tc.Circuit):
        if inplace:
            c = init_state
        else:
            c = tc.Circuit.from_qir(init_state.to_qir(), circuit_params=init_state.circuit_param)
    else:
        c = tc.Circuit(len(spin_basis), inputs=init_state)

    for ansatz_op, op_factor, name, qubit_idx_list in ansatz_op_list:
        c.exp1(*qubit_idx_list, unitary=ansatz_op, theta=dt * op_factor, name=name)
    return c


def get_ansatz(ham_terms, spin_basis, n_layers, init_state, param_ids=None):
    @jit
    def ansatz(theta):
        c = get_circuit(ham_terms, spin_basis, n_layers, init_state, theta, param_ids)
        return c.state()

    return ansatz


def get_jacobian_func(ansatz):
    return jit(tc.backend.jacfwd(ansatz, argnums=0))


def regularized_inversion(m, eps):
    evals, evecs = scipy.linalg.eigh(m)
    evals += eps * np.exp(-evals / eps)
    new_evals = 1 / evals
    return evecs @ np.diag(new_evals) @ evecs.T


def regularized_inversion2(m, eps):
    evals, evecs = scipy.linalg.eigh(m)
    mask = np.abs(evals) > 1e-10
    evals = evals[mask]
    assert np.all(evals > 0)
    new_evals = np.zeros(len(evecs))
    new_evals[mask] = 1 / evals
    return evecs @ np.diag(new_evals) @ evecs.T


def get_deriv(ansatz, jacobian_func, params, hamiltonian: np.ndarray, eps: float = 1e-5, include_phase: bool = False):
    params = tc.array_to_tensor(params)

    jacobian = tc.backend.numpy(jacobian_func(params)).astype(np.complex128)
    lhs = jacobian.conj().T @ jacobian

    psi = tc.backend.numpy(ansatz(params)).astype(np.complex128)
    hpsi = hamiltonian @ psi
    rhs = jacobian.conj().T @ hpsi

    lhs = lhs.real
    rhs = rhs.imag

    # global phase term in https://arxiv.org/pdf/1812.08767.pdf
    if include_phase:
        ovlp = jacobian.conj().T @ psi
        lhs += (ovlp.reshape(-1, 1) * ovlp.reshape(1, -1)).real
        e = psi.conj() @ hpsi
        rhs -= (e * ovlp).imag

    lhs_inverse = regularized_inversion(lhs, eps)
    theta_deriv = lhs_inverse @ rhs
    np.testing.assert_allclose(lhs @ lhs_inverse @ rhs, rhs, atol=2e2 * eps)
    np.testing.assert_allclose(theta_deriv.imag, 0)
    return theta_deriv.real.astype(np.float64)


def get_pvqd_loss_func(ansatz):
    def loss(delta_params, params, hamiltonian: np.ndarray, delta_t: float):
        ket = ansatz(params + delta_params)
        bra = ansatz(params)
        evolution = tc.backend.convert_to_tensor(tc.backend.expm(1j * delta_t * hamiltonian))
        return 1 - tc.backend.norm(bra.conj() @ (evolution @ ket)) ** 2

    loss = tc.interfaces.scipy.scipy_optimize_interface(loss)
    return loss
