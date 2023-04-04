#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import time
import logging
from functools import partial
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import tensorcircuit as tc
from renormalizer import Model, Mps, Mpo, BasisHalfSpin
from renormalizer.model.basis import BasisSet

from tencirchem.dynamic.transform import qubit_encode_op, qubit_encode_basis, get_init_circuit
from tencirchem.dynamic.time_derivative import (
    get_circuit,
    get_ansatz,
    get_jacobian_func,
    get_deriv,
    get_pvqd_loss_func,
    one_trotter_step,
)

logger = logging.getLogger(__name__)


class Ivp_Config:
    def __init__(self, method="RK45", rtol=1e-3, atol=1e-6):
        self.method = method
        self.rtol = rtol
        self.atol = atol


def evolve_exact(evals: np.ndarray, evecs: np.ndarray, init: np.ndarray, t: float):
    return evecs @ (np.diag(np.exp(-1j * t * evals)) @ (evecs.T @ init))


class TimeEvolution:
    def __init__(
        self,
        ham_terms,
        basis: List[BasisSet],
        boson_encoding: str = "gray",
        init_condition: Optional[Dict] = None,
        n_layers: int = 3,
        eps: float = 1e-5,
        property_op_dict: Dict = None,
        ref_only: bool = False,
        ivp_config: Ivp_Config = None,
    ):
        # handling defaults
        if init_condition is None:
            init_condition = {}
        if property_op_dict is None:
            property_op_dict = {}

        # setup refs
        self.model_ref = Model(basis, ham_terms)
        self.h_mpo_ref = Mpo(self.model_ref)
        self.h_ref = self.h_mpo_ref.todense()
        self.evals_ref, self.evecs_ref = scipy.linalg.eigh(self.h_ref)
        self.init_ref = Mps.hartree_product_state(self.model_ref, init_condition).todense()
        # only do ref in kernel. Could be useful for debugging
        self.ref_only = ref_only

        # setup transformed Hamiltonian
        ham_terms_spin, self.constant = qubit_encode_op(ham_terms, basis, boson_encoding)
        basis_spin: List[BasisHalfSpin] = qubit_encode_basis(basis, boson_encoding)
        # help perform some sanity checks
        self.model = Model(basis_spin, ham_terms_spin)
        self.h_mpo = Mpo(self.model)
        self.h = self.h_mpo.todense()

        # setup the ansatz
        self.init_circuit = get_init_circuit(self.model_ref, self.model, boson_encoding, init_condition)
        self.n_layers = n_layers
        self.n_params = self.n_layers * len(self.model.ham_terms)
        self.ansatz = get_ansatz(self.model.ham_terms, self.model.basis, self.n_layers, self.init_circuit)
        self.jacobian_func = get_jacobian_func(self.ansatz)

        # setup runtime components
        self.current_circuit = self.init_circuit
        self.eps = eps
        self.include_phase = False
        if ivp_config is None:
            self.ivp_config = Ivp_Config()
        else:
            self.ivp_config = ivp_config

        def scipy_deriv(t, _params):
            return get_deriv(self.ansatz, self.jacobian_func, _params, self.h, self.eps, self.include_phase)

        self.scipy_deriv = scipy_deriv

        self.pvqd_loss = get_pvqd_loss_func(self.ansatz)

        def solve_pvqd(_params, delta_t):
            hamiltonian = tc.backend.convert_to_tensor(self.h)
            loss = partial(self.pvqd_loss, params=_params, hamiltonian=hamiltonian, delta_t=delta_t)
            opt_res = minimize(loss, np.zeros_like(_params), jac=True, method="L-BFGS-B")
            return opt_res

        self.solve_pvqd = solve_pvqd

        self.property_op_dict = property_op_dict
        self.property_mat_dict = {}
        for k, op in self.property_op_dict.items():
            transformed_op, constant = qubit_encode_op(op, basis, boson_encoding)
            mat = Mpo(self.model, transformed_op).todense()
            mat += constant * np.eye(len(mat))
            mat_ref = Mpo(self.model_ref, op).todense()
            self.property_mat_dict[k] = mat, mat_ref

        # time evolution result
        self.t_list = [0]
        self.params_list = [np.zeros(self.n_params, dtype=np.float64)]
        state = self.init_circuit.state()
        self.state_list = [state]
        state_ref = self.init_ref
        self.state_ref_list = [state_ref]
        self.scipy_sol_list = []
        self._property_dict = defaultdict(list)
        self.update_property_dict(state, state_ref)

        self.wall_time_list = []

    def kernel(self, tau, algo="vqd"):
        # one step of time evolution
        if self.ref_only:
            return self.kernel_ref_only(tau)
        time0 = time.time()
        if algo == "vqd" or algo == "pvqd":
            if algo == "vqd":
                method, rtol, atol = self.ivp_config.method, self.ivp_config.rtol, self.ivp_config.atol
                scipy_sol = solve_ivp(
                    self.scipy_deriv, [self.t, self.t + tau], self.params, method=method, rtol=rtol, atol=atol
                )
                new_params = scipy_sol.y[:, -1]
            else:
                scipy_sol = self.solve_pvqd(self.params, tau)
                new_params = self.params + scipy_sol.x

            self.params_list.append(new_params)
            state = self.ansatz(self.params)
            self.scipy_sol_list.append(scipy_sol)
        else:
            assert algo == "trotter"
            self.current_circuit = one_trotter_step(
                self.model.ham_terms, self.model.basis, self.current_circuit, tau, inplace=True
            )
            shortcut = one_trotter_step(self.model.ham_terms, self.model.basis, self.state, tau)
            state = shortcut.state()

        time1 = time.time()
        self.t_list.append(self.t + tau)
        # t and params already updated
        self.state_list.append(state)
        state_ref = evolve_exact(self.evals_ref, self.evecs_ref, self.init_ref, self.t)
        self.state_ref_list.append(state_ref)
        # calculate properties
        self.update_property_dict(state, state_ref)

        self.wall_time_list.append(time1 - time0)

    def kernel_ref_only(self, tau):
        # Let's do code duplication to keep the source code simple
        self.t_list.append(self.t + tau)
        state_ref = evolve_exact(self.evals_ref, self.evecs_ref, self.init_ref, self.t)
        self.state_ref_list.append(state_ref)
        # calculate properties
        self.update_property_dict(None, state_ref)

    def update_property_dict(self, state, state_ref):
        for k, (mat, mat_ref) in self.property_mat_dict.items():
            if not self.ref_only:
                res = state.T.conj() @ (mat @ state), state_ref.T.conj() @ (mat_ref @ state_ref)
            else:
                res = state_ref.T.conj() @ (mat_ref @ state_ref)
            self._property_dict[k].append(res)

    def get_circuit(self, params, param_ids=None, compile_evolution=False):
        return get_circuit(
            self.ham_terms_spin,
            self.basis_spin,
            self.n_layers,
            self.init_circuit,
            params,
            param_ids,
            compile_evolution=compile_evolution,
        )

    @property
    def ham_terms_spin(self):
        return self.model.ham_terms

    @property
    def basis_spin(self):
        return self.model.basis

    @property
    def t(self):
        return self.t_list[-1]

    @property
    def t_array(self):
        return np.array(self.t_list)

    @property
    def params(self):
        return self.params_list[-1]

    @property
    def params_array(self):
        return np.array(self.params_list)

    @property
    def state(self):
        return self.state_list[-1]

    @property
    def state_array(self):
        return np.array(self.state_list)

    @property
    def state_ref(self):
        return self.state_ref_list[-1]

    @property
    def state_ref_array(self):
        return np.array(self.state_ref_list)

    @property
    def property_dict(self):
        return {k: np.array(v) for k, v in self._property_dict.items()}

    properties = property_dict

    @property
    def wall_time(self):
        return self.wall_time_list[-1]
