#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from typing import Tuple, List

import numpy as np
from pyscf.gto.mole import Mole
from pyscf.cc.addons import spatial2spin
from pyscf.fci import cistring
from tensorcircuit import Circuit

from tencirchem.utils.misc import rdm_mo2ao
from tencirchem.static.ucc import UCC
from tencirchem.static.ci_utils import get_ci_strings
from tencirchem.static.evolve_tensornetwork import get_circuit_givens_swap


class PUCCD(UCC):
    """
    Run paired UCC calculation.
    The interfaces are similar to :class:`UCCSD <tencirchem.UCCSD>`.
    """

    # todo: more documentation here and make the references right.
    # separate docstring examples for a variety of methods, such as energy()
    # also need to add a few comment on make_rdm1/2
    # https://arxiv.org/pdf/2002.00035.pdf
    # https://arxiv.org/pdf/1503.04878.pdf

    def __init__(
        self,
        mol: Mole,
        init_method: str = "mp2",
        active_space: Tuple[int, int] = None,
        mo_coeff: np.ndarray = None,
        engine: str = None,
        run_hf: bool = True,
        run_mp2: bool = True,
        run_ccsd: bool = True,
        run_fci: bool = True,
    ):
        r"""
        Initialize the class with molecular input.

        Parameters
        ----------
        mol: Mole
            The molecule as PySCF ``Mole`` object.
        init_method: str, optional
            How to determine the initial amplitude guess. Accepts ``"mp2"`` (default), ``"ccsd"``
            and ``"zeros"``.
        active_space: Tuple[int, int], optional
            Active space approximation. The first integer is the number of electrons and the second integer is
            the number or spatial-orbitals. Defaults to None.
        mo_coeff: np.ndarray, optional
            Molecule coefficients. If provided then RHF is skipped.
            Can be used in combination with the ``init_state`` attribute.
            Defaults to None which means RHF orbitals are used.
        engine: str, optional
            The engine to run the calculation. See :ref:`advanced:Engines` for details.
        run_hf: bool, optional
            Whether run HF for molecule orbitals. Defaults to ``True``.
        run_mp2: bool, optional
            Whether run MP2 for initial guess and energy reference. Defaults to ``True``.
        run_ccsd: bool, optional
            Whether run CCSD for initial guess and energy reference. Defaults to ``True``.
        run_fci: bool, optional
            Whether run FCI  for energy reference. Defaults to ``True``.

        See Also
        --------
        tencirchem.UCCSD
        tencirchem.KUPCCGSD
        tencirchem.UCC
        """
        super().__init__(
            mol,
            init_method,
            active_space,
            mo_coeff,
            hcb=True,
            engine=engine,
            run_hf=run_hf,
            run_mp2=run_mp2,
            run_ccsd=run_ccsd,
            run_fci=run_fci,
        )
        self.ex_ops, self.param_ids, self.init_guess = self.get_ex_ops(self.t1, self.t2)

    def get_ex1_ops(self, t1: np.ndarray = None):
        """Not applicable. Use :func:`get_ex_ops`."""
        raise NotImplementedError

    def get_ex2_ops(self, t2: np.ndarray = None):
        """Not applicable. Use :func:`get_ex_ops`."""
        raise NotImplementedError

    def get_ex_ops(self, t1: np.ndarray = None, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], List[float]]:
        """
        Get paired excitation operators for pUCCD ansatz.

        Parameters
        ----------
        t1: np.ndarray, optional
            Not used. Kept for consistency with the parent method.
        t2: np.ndarray, optional
            Not used. Kept for consistency with the parent method.

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: List[float]
            The initial guess for the parameters.

        Examples
        --------
        >>> from tencirchem import PUCCD
        >>> from tencirchem.molecule import h2
        >>> puccd = PUCCD(h2)
        >>> ex_op, param_ids, init_guess = puccd.get_ex_ops()
        >>> ex_op
        [(1, 0)]
        >>> param_ids
        [0]
        >>> init_guess  # doctest:+ELLIPSIS
        [...]
        """
        no, nv = self.no, self.nv
        if t2 is None:
            t2 = np.zeros((no, no, nv, nv))
        t2 = spatial2spin(t2)

        ex_ops = []
        ex_init_guess = []
        # to be consistent with givens rotation circuit
        for i in range(no):
            for a in range(nv - 1, -1, -1):
                ex_ops.append((no + a, i))
                ex_init_guess.append(t2[2 * i, 2 * i + 1, 2 * a, 2 * a + 1])
        return ex_ops, list(range(len(ex_ops))), ex_init_guess

    def make_rdm1(self, statevector=None, basis="AO"):
        __doc__ = super().make_rdm1.__doc__

        civector = self._statevector_to_civector(statevector)
        ci_strings = get_ci_strings(self.n_qubits, self.n_elec, self.hcb)

        n_active = self.n_qubits
        rdm1_cas = np.zeros([n_active] * 2)
        for i in range(n_active):
            bitmask = 1 << i
            arraymask = (ci_strings & bitmask) == bitmask
            value = float(civector @ (arraymask * civector))
            rdm1_cas[i, i] = 2 * value
        rdm1 = self.embed_rdm_cas(rdm1_cas)
        if basis == "MO":
            return rdm1
        else:
            return rdm_mo2ao(rdm1, self.hf.mo_coeff)

    def make_rdm2(self, statevector=None, basis="AO"):
        __doc__ = super().make_rdm2.__doc__

        civector = self._statevector_to_civector(statevector)
        ci_strings = get_ci_strings(self.n_qubits, self.n_elec, self.hcb)

        n_active = self.n_qubits
        rdm2_cas = np.zeros([n_active] * 4)
        for p in range(n_active):
            for q in range(p + 1):
                maskq = 1 << q
                maskp = 1 << p
                maskpq = maskp + maskq
                arraymask = (ci_strings & maskq) == maskq
                if p == q:
                    value = float(civector @ (arraymask * civector))
                else:
                    arraymask &= ((~ci_strings) & maskp) == maskp
                    excitation = ci_strings ^ maskpq
                    addr = cistring.strs2addr(n_active, self.n_elec // 2, excitation)
                    value = float(civector @ (arraymask * civector[addr]))

                rdm2_cas[p, q, p, q] = rdm2_cas[q, p, q, p] = value
                if p == q:
                    continue
                arraymask = (ci_strings & maskpq) == maskpq
                value = float(civector @ (arraymask * civector))

                rdm2_cas[p, p, q, q] = rdm2_cas[q, q, p, p] = 2 * value
                rdm2_cas[p, q, q, p] = rdm2_cas[q, p, p, q] = -value
        rdm2_cas *= 2
        rdm2 = self.embed_rdm_cas(rdm2_cas)
        # no need to transpose
        if basis == "MO":
            return rdm2
        else:
            return rdm_mo2ao(rdm2, self.hf.mo_coeff)

    def get_circuit(self, params=None, trotter=False, givens_swap=False) -> Circuit:
        """
        Get the circuit as TensorCircuit ``Circuit`` object.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter.
            If :func:`kernel` is not called before, the initial guess is used.
        trotter: bool, optional
            Whether Trotterize the UCC factor into Pauli strings.
            Defaults to False.
        givens_swap: bool, optional
            Whether return the circuit with Givens-Swap gates.

        Returns
        -------
        circuit: :class:`tc.Circuit`
            The quantum circuit.
        """
        if not givens_swap:
            return super().get_circuit(params, trotter=trotter)
        else:
            params = self._check_params_argument(params, strict=False)
            return get_circuit_givens_swap(params, self.n_qubits, self.n_elec, self.init_state)

    @property
    def e_puccd(self):
        """
        Returns pUCCD energy
        """
        return self.energy()
