#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from typing import Tuple, Any
import logging

import numpy as np
from pyscf.fci import cistring
from pyscf.fci.addons import des_a, cre_a, des_b, cre_b
import tensorcircuit as tc

from tencirchem.static.hamiltonian import apply_op
from tencirchem.static.ci_utils import get_init_civector

Tensor = Any

logger = logging.getLogger(__name__)


class CIvectorPySCF:
    def __init__(self, civector, n_orb, n_elec_a, n_elec_b):
        assert isinstance(civector, np.ndarray)
        self.civector = civector.reshape(-1)
        self.n_orb = n_orb
        self.n_elec_a = n_elec_a
        self.n_elec_b = n_elec_b

    def cre(self, i):
        n_elec_a, n_elec_b = self.n_elec_a, self.n_elec_b
        if i < self.n_orb:
            cre = cre_a
            n_elec_a += 1
        else:
            cre = cre_b
            n_elec_b += 1

        new_civector = cre(self.civector, self.n_orb, (self.n_elec_a, self.n_elec_b), i % self.n_orb)
        return CIvectorPySCF(new_civector, self.n_orb, n_elec_a, n_elec_b)

    def des(self, i):
        n_elec_a, n_elec_b = self.n_elec_a, self.n_elec_b
        if i < self.n_orb:
            des = des_a
            n_elec_a -= 1
        else:
            des = des_b
            n_elec_b -= 1

        new_civector = des(self.civector, self.n_orb, (self.n_elec_a, self.n_elec_b), i % self.n_orb)
        return CIvectorPySCF(new_civector, self.n_orb, n_elec_a, n_elec_b)

    def pq(self, p, q):
        return self.des(q).cre(p)

    def pqqp(self, p, q):
        return self.des(p).cre(q).des(q).cre(p)

    def pqrs(self, p, q, r, s):
        return self.des(s).des(r).cre(q).cre(p)

    def pqrssrqp(self, p, q, r, s):
        return self.des(p).des(q).cre(r).cre(s).des(s).des(r).cre(q).cre(p)


def apply_a2_pyscf(civector: CIvectorPySCF, ex_op) -> Tensor:
    if len(ex_op) == 2:
        apply_f = civector.pqqp
    else:
        assert len(ex_op) == 4
        apply_f = civector.pqrssrqp
    civector1 = apply_f(*ex_op)
    civector2 = apply_f(*reversed(ex_op))
    return -civector1.civector - civector2.civector


def apply_a_pyscf(civector: CIvectorPySCF, ex_op) -> Tensor:
    if len(ex_op) == 2:
        apply_func = civector.pq
    else:
        assert len(ex_op) == 4
        apply_func = civector.pqrs
    civector1 = apply_func(*ex_op)
    civector2 = apply_func(*reversed(ex_op))
    return civector1.civector - civector2.civector


def evolve_excitation_pyscf(civector: Tensor, ex_op, n_orb, n_elec, theta) -> Tensor:
    ket = CIvectorPySCF(civector, n_orb, n_elec // 2, n_elec // 2)
    aket = apply_a_pyscf(ket, ex_op)
    a2ket = apply_a2_pyscf(ket, ex_op)
    return civector + (1 - np.cos(theta)) * a2ket + np.sin(theta) * aket


def get_civector_pyscf(params, n_qubits, n_elec, ex_ops, param_ids, hcb=False, init_state=None):
    assert not hcb
    n_orb = n_qubits // 2
    num_strings = cistring.num_strings(n_orb, n_elec // 2) ** 2

    if init_state is None:
        civector = get_init_civector(num_strings)
    else:
        civector = tc.backend.convert_to_tensor(init_state)

    civector = tc.backend.numpy(civector)

    for ex_op, param_id in zip(ex_ops, param_ids):
        theta = params[param_id]
        civector = evolve_excitation_pyscf(civector, ex_op, n_orb, n_elec, theta)

    return civector.reshape(-1)


def get_energy_and_grad_pyscf(
    params, hamiltonian, n_qubits, n_elec, ex_ops: Tuple, param_ids: Tuple, hcb: bool = False, init_state=None
):
    params = tc.backend.numpy(params)
    ket = get_civector_pyscf(params, n_qubits, n_elec, ex_ops, param_ids, hcb, init_state)
    bra = tc.backend.numpy(apply_op(hamiltonian, ket))
    energy = bra @ ket

    gradients_beforesum = _get_gradients_pyscf(bra, ket, params, n_qubits, n_elec, ex_ops, param_ids, hcb)

    gradients = np.zeros(params.shape)
    for grad, param_id in zip(gradients_beforesum, param_ids):
        gradients[param_id] += grad

    return energy, 2 * gradients


def _get_gradients_pyscf(bra, ket, params, n_qubits, n_elec, ex_ops, param_ids, hcb):
    assert not hcb

    n_orb = n_qubits // 2

    gradients_beforesum = []
    for param_id, ex_op in reversed(list(zip(param_ids, ex_ops))):
        theta = params[param_id]
        bra = evolve_excitation_pyscf(bra, ex_op, n_orb, n_elec, -theta)
        ket = evolve_excitation_pyscf(ket, ex_op, n_orb, n_elec, -theta)
        ket_pyscf = CIvectorPySCF(ket, n_orb, n_elec // 2, n_elec // 2)
        fket = apply_a_pyscf(ket_pyscf, ex_op)
        grad = bra @ fket
        gradients_beforesum.append(grad)
    gradients_beforesum = list(reversed(gradients_beforesum))
    gradients_beforesum = np.array(gradients_beforesum)

    return gradients_beforesum


def apply_excitation_pyscf(civector, n_qubits, n_elec, f_idx, hcb):
    assert not hcb
    civector_pyscf = CIvectorPySCF(civector, n_qubits // 2, n_elec // 2, n_elec // 2)
    return apply_a_pyscf(civector_pyscf, f_idx)
