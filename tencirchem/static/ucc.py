#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from functools import partial
from itertools import product
from collections import defaultdict
from time import time
import logging
from typing import Any, Tuple, Callable, List, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import comb
import pandas as pd
from openfermion import jordan_wigner, FermionOperator, QubitOperator
from pyscf.gto.mole import Mole
from pyscf.scf import RHF
from pyscf.cc.addons import spatial2spin
from pyscf.mcscf import CASCI
from pyscf import fci
import tensorcircuit as tc

from tencirchem.constants import DISCARD_EPS
from tencirchem.molecule import _Molecule
from tencirchem.utils.misc import reverse_qop_idx, scipy_opt_wrap, rdm_mo2ao, canonical_mo_coeff
from tencirchem.utils.circuit import get_circuit_dataframe
from tencirchem.static.engine_ucc import (
    get_civector,
    get_statevector,
    get_energy,
    get_energy_and_grad,
    apply_excitation,
    translate_init_state,
)
from tencirchem.static.hamiltonian import (
    get_integral_from_hf,
    get_h_from_integral,
    get_hop_from_integral,
    get_hop_hcb_from_integral,
)
from tencirchem.static.ci_utils import get_ci_strings, get_ex_bitstring, get_addr, get_init_civector
from tencirchem.static.evolve_tensornetwork import get_circuit


logger = logging.getLogger(__name__)


Tensor = Any


class UCC:
    """
    Base class for :class:`UCCSD`.
    """

    @classmethod
    def from_integral(
        cls, int1e: np.ndarray, int2e: np.ndarray, n_elec: int, e_core: float = 0, ovlp: np.ndarray = None, **kwargs
    ):
        """
        Construct UCC classes from electron integrals.

        Parameters
        ----------
        int1e: np.ndarray
            One-body integral in spatial orbital.
        int2e: np.ndarray
            Two-body integral, in spatial orbital, chemists' notation, and without considering symmetry.
        n_elec: int
            The number of electrons
        e_core: float, optional
            The nuclear energy or core energy if active space approximation is involved.
            Defaults to 0.
        ovlp: np.ndarray
            The overlap integral. Defaults to ``None`` and identity matrix is used.
        kwargs:
            Other arguments to be passed to the :func:`__init__` function such as ``engine``.

        Returns
        -------
        ucc: :class:`UCC`
             A UCC instance
        """
        if isinstance(n_elec, tuple):
            if len(n_elec) != 2 or n_elec[0] != n_elec[1]:
                raise ValueError(f"Incompatible n_elec: {n_elec}")
            n_elec = n_elec[0] + n_elec[1]
        m = _Molecule(int1e, int2e, n_elec, e_core, ovlp)
        return cls(m, **kwargs)

    @classmethod
    def as_pyscf_solver(cls, config_function: Callable = None, **kwargs):
        """
        Converts the ``UCC`` class to a PySCF FCI solver.

        Parameters
        ----------
        config_function: callable
            A function to configure the ``UCC`` instance.
            Accepts the ``UCC`` instance and modifies it inplace before :func:`kernel` is called.
        kwargs
            Other arguments to be passed to the :func:`__init__` function such as ``engine``.

        Returns
        -------
        FCISolver

        Examples
        --------
        >>> from pyscf.mcscf import CASSCF
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h8
        >>> # normal PySCF workflow
        >>> hf = h8.HF()
        >>> round(hf.kernel(), 8)
        -4.14961853
        >>> casscf = CASSCF(hf, 2, 2)
        >>> # set the FCI solver for CASSCF to be UCCSD
        >>> casscf.fcisolver = UCCSD.as_pyscf_solver()
        >>> round(casscf.kernel()[0], 8)
        -4.16647335
        """

        class FakeFCISolver:
            def __init__(self):
                self.instance: UCC = None
                self.config_function = config_function
                self.instance_kwargs = kwargs
                for arg in ["run_ccsd", "run_fci"]:
                    # keep MP2 for initial guess
                    self.instance_kwargs[arg] = False

            def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
                self.instance = cls.from_integral(h1, h2, nelec, **self.instance_kwargs)
                if self.config_function is not None:
                    self.config_function(self.instance)
                e = self.instance.kernel()
                return e + ecore, self.instance.params

            def make_rdm1(self, params, norb, nelec):
                civector = self.instance.civector(params)
                return self.instance.make_rdm1(civector)

            def make_rdm12(self, params, norb, nelec):
                civector = self.instance.civector(params)
                rdm1 = self.instance.make_rdm1(civector)
                rdm2 = self.instance.make_rdm2(civector)
                return rdm1, rdm2

            def spin_square(self, params, norb, nelec):
                return 0, 1

        return FakeFCISolver()

    def __init__(
        self,
        mol: Mole,
        init_method="mp2",
        active_space=None,
        mo_coeff=None,
        hcb=False,
        engine=None,
        run_hf=True,
        run_mp2=True,
        run_ccsd=True,
        run_fci=True,
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
        hcb: bool, optional
            Whether force electrons to pair as hard-core boson (HCB). Default to False.
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
        tencirchem.PUCCD
        """
        # process mol
        if isinstance(mol, _Molecule):
            self.mol = mol
        else:
            # to set verbose = 0
            self.mol = mol.copy()
            if mo_coeff is None:
                self.mol.symmetry = True
            self.mol.build()
        if active_space is None:
            active_space = (mol.nelectron, int(mol.nao))

        self.hcb = hcb
        self.n_qubits = 2 * active_space[1]
        if hcb:
            self.n_qubits //= 2

        # process activate space
        self.active_space = active_space
        self.n_elec = active_space[0]
        self.active = active_space[1]
        self.inactive_occ = mol.nelectron // 2 - active_space[0] // 2
        self.inactive_vir = mol.nao - active_space[1] - self.inactive_occ
        frozen_idx = list(range(self.inactive_occ)) + list(range(mol.nao - self.inactive_vir, mol.nao))

        # process backend
        self._check_engine(engine)

        if engine is None:
            # no need to be too precise
            if self.n_qubits <= 16:
                engine = "civector"
            else:
                engine = "civector-large"
        self.engine = engine

        # classical quantum chemistry
        # hf
        self.mol.verbose = 0
        self.hf = RHF(self.mol)
        # avoid serialization warnings for `_Molecule`
        self.hf.chkfile = None
        if run_hf:
            # run this even when ``mo_coeff is not None`` because MP2 and CCSD
            # reference energy might be desired
            self.e_hf = self.hf.kernel()
            self.hf.mo_coeff = canonical_mo_coeff(self.hf.mo_coeff)
        else:
            self.e_hf = None
            # otherwise, can't run casci.get_h2eff() based on HF
            self.hf._eri = mol.intor("int2e", aosym="s8")
            if mo_coeff is None:
                raise ValueError("Must provide MO coefficient if HF is skipped")

        # mp2
        if run_mp2:
            mp2 = self.hf.MP2()
            if frozen_idx:
                mp2.frozen = frozen_idx
            e_corr_mp2, mp2_t2 = mp2.kernel()
            self.e_mp2 = self.e_hf + e_corr_mp2
        else:
            self.e_mp2 = None
            mp2_t2 = None
            if init_method is not None and init_method.lower() == "mp2":
                raise ValueError("Must run MP2 to use MP2 as the initial guess method")

        # ccsd
        if run_ccsd:
            ccsd = self.hf.CCSD()
            if frozen_idx:
                ccsd.frozen = frozen_idx
            e_corr_ccsd, ccsd_t1, ccsd_t2 = ccsd.kernel()
            self.e_ccsd = self.e_hf + e_corr_ccsd
        else:
            self.e_ccsd = None
            ccsd_t1 = ccsd_t2 = None
            if init_method is not None and init_method.lower() == "ccsd":
                raise ValueError("Must run CCSD to use CCSD as the initial guess method")

        # MP2 and CCSD rely on canonical HF orbitals but FCI doesn't
        # so set custom mo_coeff after MP2 and CCSD and before FCI
        if mo_coeff is not None:
            # use user defined coefficient
            self.hf.mo_coeff = canonical_mo_coeff(mo_coeff)

        # fci
        if run_fci:
            fci = CASCI(self.hf, self.active_space[1], self.active_space[0])
            fci.max_memory = 32000
            res = fci.kernel()
            self.e_fci = res[0]
            self.civector_fci = res[2].ravel()
        else:
            self.e_fci = None
            self.civector_fci = None

        self.e_nuc = mol.energy_nuc()

        # Hamiltonian related
        self.hamiltonian_lib = {}
        self.int1e = self.int2e = None
        # e_core includes nuclear repulsion energy
        self.hamiltonian, self.e_core, _ = self._get_hamiltonian_and_core(self.engine)

        # initial guess
        self.t1 = np.zeros([self.no, self.nv])
        self.t2 = np.zeros([self.no, self.no, self.nv, self.nv])
        if init_method is None or init_method in ["zeros", "zero"]:
            pass
        elif init_method.lower() == "ccsd":
            self.t1, self.t2 = ccsd_t1, ccsd_t2
        elif init_method.lower() == "fe":
            self.t2 = compute_fe_t2(self.no, self.nv, self.int1e, self.int2e)
        elif init_method.lower() == "mp2":
            self.t2 = mp2_t2
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        # circuit related
        self._init_state = None
        self.ex_ops = None
        self._param_ids = None
        self.init_guess = None

        # optimization related
        self.scipy_minimize_options = None
        # optimization result
        self.opt_res = None
        # for manually set
        self._params = None

    def kernel(self) -> float:
        """
        The kernel to perform the VQE algorithm.
        The L-BFGS-B method in SciPy is used for optimization
        and configuration is possible by setting the ``self.scipy_minimize_options`` attribute.

        Returns
        -------
        e: float
            The optimized energy
        """
        assert len(self.param_ids) == len(self.ex_ops)

        energy_and_grad, stating_time = self.get_opt_function(with_time=True)

        if self.init_guess is None:
            self.init_guess = np.zeros(self.n_params)

        # optimization options
        if self.scipy_minimize_options is None:
            # quite strict
            options = {"ftol": 1e1 * np.finfo(tc.rdtypestr).eps, "gtol": 1e2 * np.finfo(tc.rdtypestr).eps}
        else:
            options = self.scipy_minimize_options

        logger.info("Begin optimization")

        time1 = time()
        opt_res = minimize(energy_and_grad, x0=self.init_guess, jac=True, method="L-BFGS-B", options=options)
        time2 = time()

        if not opt_res.success:
            logger.warning("Optimization failed. See `.opt_res` for details.")

        opt_res["staging_time"] = stating_time
        opt_res["opt_time"] = time2 - time1
        opt_res["init_guess"] = self.init_guess
        opt_res["e"] = float(opt_res.fun)
        self.opt_res = opt_res
        # prepare for future modification
        self.params = opt_res.x.copy()
        return opt_res.e

    def get_opt_function(self, with_time: bool = False) -> Union[Callable, Tuple[Callable, float]]:
        """
        Returns the cost function in SciPy format for optimization.
        The gradient is included.
        Basically a wrapper to :func:`energy_and_grad`.

        Parameters
        ----------
        with_time: bool, optional
            Whether return staging time. Defaults to False.

        Returns
        -------
        opt_function: Callable
            The optimization cost function in SciPy format.
        time: float
            Staging time. Returned when ``with_time`` is set to ``True``.
        """
        energy_and_grad = scipy_opt_wrap(partial(self.energy_and_grad, engine=self.engine))

        time1 = time()
        if tc.backend.name == "jax":
            logger.info("JIT compiling the circuit")
            _ = energy_and_grad(np.zeros(self.n_params))
            logger.info("Circuit JIT compiled")
        time2 = time()
        if with_time:
            return energy_and_grad, time2 - time1
        return energy_and_grad

    def _check_params_argument(self, params, strict=True):
        if params is None:
            if self.params is not None:
                params = self.params
            else:
                if strict:
                    raise ValueError("Run the `.kernel` method to determine the parameters first")
                else:
                    if self.init_guess is not None:
                        params = self.init_guess
                    else:
                        params = np.zeros(self.n_params)

        if len(params) != self.n_params:
            raise ValueError(f"Incompatible parameter shape. {self.n_params} is desired. Got {len(params)}")
        return tc.backend.convert_to_tensor(params).astype(tc.rdtypestr)

    def _check_engine(self, engine):
        supported_engine = [None, "tensornetwork", "statevector", "civector", "civector-large", "pyscf"]
        if not engine in supported_engine:
            raise ValueError(f"Engine '{engine}' not supported")

    def _sanity_check(self):
        if self.ex_ops is None or self.param_ids is None:
            raise ValueError("`ex_ops` or `param_ids` not defined")
        if self.param_ids is not None and (len(self.ex_ops) != len(self.param_ids)):
            raise ValueError(
                f"Excitation operator size {len(self.ex_ops)} and parameter size {len(self.param_ids)} do not match"
            )

    def civector(self, params: Tensor = None, engine: str = None) -> Tensor:
        """
        Evaluate the configuration interaction (CI) vector.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        civector: Tensor
            Corresponding CI vector

        See Also
        --------
        statevector: Evaluate the circuit state vector.
        energy: Evaluate the total energy.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.civector([0, 0])  # HF state
        array([1., 0., 0., 0.])
        """
        self._sanity_check()
        params = self._check_params_argument(params)
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
        civector = get_civector(
            params, self.n_qubits, self.n_elec, self.ex_ops, self.param_ids, self.hcb, self.init_state, engine
        )
        return civector

    def get_ci_strings(self, strs2addr: bool = False) -> np.ndarray:
        """
        Get the CI bitstrings for all configurations in the CI vector.

        Parameters
        ----------
        strs2addr: bool, optional.
            Whether return the reversed mapping for one spin sector. Defaults to ``False``.

        Returns
        -------
        cistrings: np.ndarray
            The CI bitstrings.
        string_addr: np.ndarray
            The address of the string in one spin sector. Returned when ``strs2addr`` is set to ``True``.

        Examples
        --------
        >>> from tencirchem import UCCSD, PUCCD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.get_ci_strings()
        array([ 5,  6,  9, 10], dtype=uint64)
        >>> [f"{bin(i)[2:]:0>4}" for i in uccsd.get_ci_strings()]
        ['0101', '0110', '1001', '1010']
        >>> uccsd.get_ci_strings(True)[1]  # only one spin sector
        array([0, 0, 1, 0], dtype=uint64)
        """
        return get_ci_strings(self.n_qubits, self.n_elec, self.hcb, strs2addr=strs2addr)

    def get_addr(self, bitstring: str) -> int:
        """
        Get the address (index) of a CI bitstring in the CI vector.

        Parameters
        ----------
        bitstring: str
            The bitstring such as ``"0101"``.

        Returns
        -------
        address: int
            The bitstring address.

        Examples
        --------
        >>> from tencirchem import UCCSD, PUCCD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.get_addr("0101")  # the HF state
        0
        >>> uccsd.get_addr("1010")
        3
        >>> puccd = PUCCD(h2)
        >>> puccd.get_addr("01")  # the HF state
        0
        >>> puccd.get_addr("10")
        1
        """
        _, strs2addr = self.get_ci_strings(strs2addr=True)
        return int(get_addr(int(bitstring, base=2), self.n_qubits, self.n_elec, strs2addr, self.hcb))

    def statevector(self, params: Tensor = None, engine: str = None) -> Tensor:
        """
        Evaluate the circuit state vector.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        statevector: Tensor
            Corresponding state vector

        See Also
        --------
        civector: Evaluate the configuration interaction (CI) vector.
        energy: Evaluate the total energy.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.statevector([0, 0])  # HF state
        array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        """
        self._sanity_check()
        params = self._check_params_argument(params)
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
        statevector = get_statevector(
            params, self.n_qubits, self.n_elec, self.ex_ops, self.param_ids, self.hcb, self.init_state, engine
        )
        return statevector

    def _get_hamiltonian_and_core(self, engine):
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
            hamiltonian = self.hamiltonian
            e_core = self.e_core
        else:
            if engine.startswith("civector") or engine == "pyscf":
                htype = "fcifunc"
            else:
                assert engine in ["tensornetwork", "statevector"]
                htype = "sparse"
            hamiltonian = self.hamiltonian_lib.get(htype)
            if hamiltonian is None:
                if self.int1e is None:
                    self.int1e, self.int2e, e_core = get_integral_from_hf(self.hf, self.active_space)
                else:
                    e_core = self.e_core
                hamiltonian = get_h_from_integral(self.int1e, self.int2e, self.n_elec, self.hcb, htype)
                self.hamiltonian_lib[htype] = hamiltonian
            else:
                e_core = self.e_core
        return hamiltonian, e_core, engine

    def energy(self, params: Tensor = None, engine: str = None) -> float:
        """
        Evaluate the total energy.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        energy: float
            Total energy

        See Also
        --------
        civector: Get the configuration interaction (CI) vector.
        statevector: Evaluate the circuit state vector.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> round(uccsd.energy([0, 0]), 8)  # HF state
        -1.11670614
        """
        self._sanity_check()
        params = self._check_params_argument(params)
        if params is self.params and self.opt_res is not None:
            return self.opt_res.e
        hamiltonian, _, engine = self._get_hamiltonian_and_core(engine)
        e = get_energy(
            params,
            hamiltonian,
            self.n_qubits,
            self.n_elec,
            self.ex_ops,
            self.param_ids,
            self.hcb,
            self.init_state,
            engine,
        )
        return float(e) + self.e_core

    def energy_and_grad(self, params: Tensor = None, engine: str = None) -> Tuple[float, Tensor]:
        """
        Evaluate the total energy and parameter gradients.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        energy: float
            Total energy
        grad: Tensor
            The parameter gradients

        See Also
        --------
        civector: Get the configuration interaction (CI) vector.
        statevector: Evaluate the circuit state vector.
        energy: Evaluate the total energy.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> e, g = uccsd.energy_and_grad([0, 0])
        >>> round(e, 8)
        -1.11670614
        >>> g  # doctest:+ELLIPSIS
        array([..., ...])
        """
        self._sanity_check()
        params = self._check_params_argument(params)
        hamiltonian, _, engine = self._get_hamiltonian_and_core(engine)
        e, g = get_energy_and_grad(
            params,
            hamiltonian,
            self.n_qubits,
            self.n_elec,
            self.ex_ops,
            self.param_ids,
            self.hcb,
            self.init_state,
            engine,
        )
        return float(e + self.e_core), tc.backend.numpy(g)

    def apply_excitation(self, state: Tensor, ex_op: Tuple, engine: str = None) -> Tensor:
        """
        Apply a given excitation operator to a given state.

        Parameters
        ----------
        state: Tensor
            The input state in statevector or CI vector form.
        ex_op: tuple of ints
            The excitation operator.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.
        Returns
        -------
        tensor: Tensor
            The resulting tensor.

        Examples
        --------
        >>> from tencirchem import UCC
        >>> from tencirchem.molecule import h2
        >>> ucc = UCC(h2)
        >>> ucc.apply_excitation([1, 0, 0, 0], (3, 1, 0, 2))
        array([0, 0, 0, 1])
        """
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
        return apply_excitation(state, self.n_qubits, self.n_elec, ex_op, hcb=self.hcb, engine=engine)

    def _statevector_to_civector(self, statevector=None):
        if statevector is None:
            civector = self.civector()
        else:
            if len(statevector) == self.statevector_size:
                ci_strings = self.get_ci_strings()
                civector = statevector[ci_strings]
            else:
                if len(statevector) == self.civector_size:
                    civector = statevector
                else:
                    raise ValueError(f"Incompatible statevector size: {len(statevector)}")

        civector = tc.backend.numpy(tc.backend.convert_to_tensor(civector))
        return civector

    def make_rdm1(self, statevector: Tensor = None, basis: str = "AO") -> np.ndarray:
        r"""
        Evaluate the spin-traced one-body reduced density matrix (1RDM).

        .. math::

            \textrm{1RDM}[p,q] = \langle p_{\alpha}^\dagger q_{\alpha} \rangle
                + \langle p_{\beta}^\dagger q_{\beta} \rangle

        If active space approximation is employed, returns the full RDM of all orbitals.

        Parameters
        ----------
        statevector: Tensor, optional
            Custom system state. Could be CI vector or state vector.
            Defaults to None, which uses the optimized state by :func:`civector`.

        basis: str, optional
            One of ``"AO"`` or ``"MO"``. Defaults to ``"AO"``, which is for consistency with PySCF.

        Returns
        -------
        rdm1: np.ndarray
            The spin-traced one-body RDM.

        See Also
        --------
        make_rdm2: Evaluate the spin-traced two-body reduced density matrix (2RDM).
        """
        assert not self.hcb
        civector = self._statevector_to_civector(statevector).astype(np.float64)

        rdm1_cas = fci.direct_spin1.make_rdm1(civector, self.n_qubits // 2, self.n_elec)

        rdm1 = self.embed_rdm_cas(rdm1_cas)

        if basis == "MO":
            return rdm1
        else:
            return rdm_mo2ao(rdm1, self.hf.mo_coeff)

    def make_rdm2(self, statevector: Tensor = None, basis: str = "AO") -> np.ndarray:
        r"""
        Evaluate the spin-traced two-body reduced density matrix (2RDM).

        .. math::

            \begin{aligned}
                \textrm{2RDM}[p,q,r,s] & = \langle p_{\alpha}^\dagger r_{\alpha}^\dagger
                s_{\alpha}  q_{\alpha} \rangle
                   + \langle p_{\beta}^\dagger r_{\alpha}^\dagger s_{\alpha}  q_{\beta} \rangle \\
                   & \quad + \langle p_{\alpha}^\dagger r_{\beta}^\dagger s_{\beta}  q_{\alpha} \rangle
                   + \langle p_{\beta}^\dagger r_{\beta}^\dagger s_{\beta}  q_{\beta} \rangle
            \end{aligned}

        If active space approximation is employed, returns the full RDM of all orbitals.

        Parameters
        ----------
        statevector: Tensor, optional
            Custom system state. Could be CI vector or state vector.
            Defaults to None, which uses the optimized state by :func:`civector`.

        basis: str, optional
            One of ``"AO"`` or ``"MO"``. Defaults to ``"AO"``, which is for consistency with PySCF.

        Returns
        -------
        rdm2: np.ndarray
            The spin-traced two-body RDM.

        See Also
        --------
        make_rdm1: Evaluate the spin-traced one-body reduced density matrix (1RDM).

        Examples
        --------
        >>> import numpy as np
        >>> from tencirchem import UCC
        >>> from tencirchem.molecule import h2
        >>> ucc = UCC(h2)
        >>> state = [1, 0, 0, 0]  ## HF state
        >>> rdm1 = ucc.make_rdm1(state, basis="MO")
        >>> rdm2 = ucc.make_rdm2(state, basis="MO")
        >>> e_hf = ucc.int1e.ravel() @ rdm1.ravel() + 1/2 * ucc.int2e.ravel() @ rdm2.ravel()
        >>> np.testing.assert_allclose(e_hf + ucc.e_nuc, ucc.e_hf, atol=1e-10)
        """
        assert not self.hcb
        civector = self._statevector_to_civector(statevector).astype(np.float64)

        rdm2_cas = fci.direct_spin1.make_rdm12(civector.astype(np.float64), self.n_qubits // 2, self.n_elec)[1]

        rdm2 = self.embed_rdm_cas(rdm2_cas)

        if basis == "MO":
            return rdm2
        else:
            return rdm_mo2ao(rdm2, self.hf.mo_coeff)

    def embed_rdm_cas(self, rdm_cas):
        """
        Embed CAS RDM into RDM of the whole system
        """
        if self.inactive_occ == 0 and self.inactive_vir == 0:
            # active space approximation not employed
            return rdm_cas
        # slice of indices in rdm corresponding to cas
        slice_cas = slice(self.inactive_occ, self.inactive_occ + len(rdm_cas))
        if rdm_cas.ndim == 2:
            rdm1_cas = rdm_cas
            rdm1 = np.zeros((self.mol.nao, self.mol.nao))
            for i in range(self.inactive_occ):
                rdm1[i, i] = 2
            rdm1[slice_cas, slice_cas] = rdm1_cas
            return rdm1
        else:
            rdm2_cas = rdm_cas
            # active space approximation employed
            rdm1 = self.make_rdm1(basis="MO")
            rdm1_cas = rdm1[slice_cas, slice_cas]
            rdm2 = np.zeros((self.mol.nao, self.mol.nao, self.mol.nao, self.mol.nao))
            rdm2[slice_cas, slice_cas, slice_cas, slice_cas] = rdm2_cas
            for i in range(self.inactive_occ):
                for j in range(self.inactive_occ):
                    rdm2[i, i, j, j] += 4
                    rdm2[i, j, j, i] -= 2
                rdm2[i, i, slice_cas, slice_cas] = rdm2[slice_cas, slice_cas, i, i] = 2 * rdm1_cas
                rdm2[i, slice_cas, slice_cas, i] = rdm2[slice_cas, i, i, slice_cas] = -rdm1_cas
            return rdm2

    def get_ex_ops(self, t1: np.ndarray = None, t2: np.ndarray = None):
        """Virtual method to be implemented"""
        raise NotImplementedError

    def get_ex1_ops(self, t1: np.ndarray = None) -> Tuple[List[Tuple], List[int], List[float]]:
        """
        Get one-body excitation operators.

        Parameters
        ----------
        t1: np.ndarray, optional
            Initial one-body amplitudes based on e.g. CCSD

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: List[float]
            The initial guess for the parameters.

        See Also
        --------
        get_ex2_ops: Get two-body excitation operators.
        get_ex_ops: Get one-body and two-body excitation operators for UCCSD ansatz.
        """
        # single excitations
        no, nv = self.no, self.nv
        if t1 is None:
            t1 = self.t1

        if t1.shape == (self.no, self.nv):
            t1 = spatial2spin(t1)
        else:
            assert t1.shape == (2 * self.no, 2 * self.nv)

        ex1_ops = []
        # unique parameters. -1 is a place holder
        ex1_param_ids = [-1]
        ex1_init_guess = []
        for i in range(no):
            for a in range(nv):
                # alpha to alpha
                ex_op_a = (2 * no + nv + a, no + nv + i)
                # beta to beta
                ex_op_b = (no + a, i)
                ex1_ops.extend([ex_op_a, ex_op_b])
                ex1_param_ids.extend([ex1_param_ids[-1] + 1] * 2)
                ex1_init_guess.append(t1[i, a])

        return ex1_ops, ex1_param_ids[1:], ex1_init_guess

    def get_ex2_ops(self, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], List[float]]:
        """
        Get two-body excitation operators.

        Parameters
        ----------
        t2: np.ndarray, optional
            Initial two-body amplitudes based on e.g. MP2

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: List[float]
            The initial guess for the parameters.

        See Also
        --------
        get_ex1_ops: Get one-body excitation operators.
        get_ex_ops: Get one-body and two-body excitation operators for UCCSD ansatz.
        """

        # t2 in oovv 1212 format
        no, nv = self.no, self.nv
        if t2 is None:
            t2 = self.t2

        if t2.shape == (self.no, self.no, self.nv, self.nv):
            t2 = spatial2spin(t2)
        else:
            assert t2.shape == (2 * self.no, 2 * self.no, 2 * self.nv, 2 * self.nv)

        def alpha_o(_i):
            return no + nv + _i

        def alpha_v(_i):
            return 2 * no + nv + _i

        def beta_o(_i):
            return _i

        def beta_v(_i):
            return no + _i

        # double excitations
        ex_ops = []
        ex2_param_ids = [-1]
        ex2_init_guess = []
        # 2 alphas or 2 betas
        for i in range(no):
            for j in range(i):
                for a in range(nv):
                    for b in range(a):
                        # i correspond to a and j correspond to b, as in PySCF convention
                        # otherwise the t2 amplitude has incorrect phase
                        # 2 alphas
                        ex_op_aa = (alpha_v(b), alpha_v(a), alpha_o(i), alpha_o(j))
                        # 2 betas
                        ex_op_bb = (beta_v(b), beta_v(a), beta_o(i), beta_o(j))
                        ex_ops.extend([ex_op_aa, ex_op_bb])
                        ex2_param_ids.extend([ex2_param_ids[-1] + 1] * 2)
                        ex2_init_guess.append(t2[2 * i, 2 * j, 2 * a, 2 * b])
        assert len(ex_ops) == 2 * (no * (no - 1) / 2) * (nv * (nv - 1) / 2)
        # 1 alpha + 1 beta
        for i in range(no):
            for j in range(i + 1):
                for a in range(nv):
                    for b in range(a + 1):
                        # i correspond to a and j correspond to b, as in PySCF convention
                        # otherwise the t2 amplitude has incorrect phase
                        if i == j and a == b:
                            # paired
                            ex_op_ab = (beta_v(a), alpha_v(a), alpha_o(i), beta_o(i))
                            ex_ops.append(ex_op_ab)
                            ex2_param_ids.append(ex2_param_ids[-1] + 1)
                            ex2_init_guess.append(t2[2 * i, 2 * i + 1, 2 * a, 2 * a + 1])
                            continue
                        # simple reflection
                        ex_op_ab1 = (beta_v(b), alpha_v(a), alpha_o(i), beta_o(j))
                        ex_op_ab2 = (alpha_v(b), beta_v(a), beta_o(i), alpha_o(j))
                        ex_ops.extend([ex_op_ab1, ex_op_ab2])
                        ex2_param_ids.extend([ex2_param_ids[-1] + 1] * 2)
                        ex2_init_guess.append(t2[2 * i, 2 * j + 1, 2 * a, 2 * b + 1])
                        if (i != j) and (a != b):
                            # exchange alpha and beta
                            ex_op_ab3 = (beta_v(a), alpha_v(b), alpha_o(i), beta_o(j))
                            ex_op_ab4 = (alpha_v(a), beta_v(b), beta_o(i), alpha_o(j))
                            ex_ops.extend([ex_op_ab3, ex_op_ab4])
                            ex2_param_ids.extend([ex2_param_ids[-1] + 1] * 2)
                            ex2_init_guess.append(t2[2 * i, 2 * j + 1, 2 * b, 2 * a + 1])

        return ex_ops, ex2_param_ids[1:], ex2_init_guess

    @property
    def e_ucc(self) -> float:
        """
        Returns UCC energy
        """
        return self.energy()

    def print_ansatz(self):
        df_dict = {
            "#qubits": [self.n_qubits],
            "#params": [self.n_params],
            "#excitations": [len(self.ex_ops)],
        }
        if self.init_state is None:
            df_dict["initial condition"] = "RHF"
        else:
            df_dict["initial condition"] = "custom"
        print(pd.DataFrame(df_dict).to_string(index=False))

    def get_circuit(
        self, params: Tensor = None, decompose_multicontrol: bool = False, trotter: bool = False
    ) -> tc.Circuit:
        """
        Get the circuit as TensorCircuit ``Circuit`` object

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter.
            If :func:`kernel` is not called before, the initial guess is used.
        decompose_multicontrol: bool, optional
            Whether decompose the Multicontrol gate in the circuit into CNOT gates.
            Defaults to False.
        trotter: bool, optional
            Whether Trotterize the UCC factor into Pauli strings.
            Defaults to False.

        Returns
        -------
        circuit: :class:`tc.Circuit`
            The quantum circuit.
        """
        if self.ex_ops is None:
            raise ValueError("Excitation operators not defined")
        params = self._check_params_argument(params, strict=False)
        return get_circuit(
            params,
            self.n_qubits,
            self.n_elec,
            self.ex_ops,
            self.param_ids,
            self.hcb,
            self.init_state,
            decompose_multicontrol=decompose_multicontrol,
            trotter=trotter,
        )

    def print_circuit(self):
        """
        Prints the circuit information. If you wish to print the circuit diagram,
        use :func:`get_circuit` and then call ``draw()`` such as ``print(ucc.get_circuit().draw())``.
        """
        c = self.get_circuit()
        df = get_circuit_dataframe(c)

        def format_flop(f):
            return f"{f:.3e}"

        formatters = {"flop": format_flop}
        print(df.to_string(index=False, formatters=formatters))

    def get_init_state_dataframe(self, coeff_epsilon: float = DISCARD_EPS) -> pd.DataFrame:
        """
        Returns initial state information dataframe.

        Parameters
        ----------
        coeff_epsilon: float, optional
            The threshold to screen out states with small coefficients.
            Defaults to 1e-12.

        Returns
        -------
        pd.DataFrame

        See Also
        --------
        init_state: The circuit initial state before applying the excitation operators.

        Examples
        --------
        >>> from tencirchem import UCC
        >>> from tencirchem.molecule import h2
        >>> ucc = UCC(h2)
        >>> ucc.init_state = [0.707, 0, 0, 0.707]
        >>> ucc.get_init_state_dataframe()   # doctest: +NORMALIZE_WHITESPACE
             configuration  coefficient
        0          0101        0.707
        1          1010        0.707
        """
        columns = ["configuration", "coefficient"]
        if self.init_state is None:
            init_state = get_init_civector(self.civector_size)
        else:
            init_state = self.init_state
        ci_strings = self.get_ci_strings()
        ci_coeffs = translate_init_state(init_state, self.n_qubits, ci_strings)
        data_list = []
        for ci_string, coeff in zip(ci_strings, ci_coeffs):
            if np.abs(coeff) < coeff_epsilon:
                continue
            ci_string = bin(ci_string)[2:]
            ci_string = "0" * (self.n_qubits - len(ci_string)) + ci_string
            data_list.append((ci_string, coeff))
        return pd.DataFrame(data_list, columns=columns)

    def print_init_state(self):
        print(self.get_init_state_dataframe().to_string())

    def get_excitation_dataframe(self) -> pd.DataFrame:
        columns = ["excitation", "configuration", "parameter", "initial guess"]
        if self.ex_ops is None:
            return pd.DataFrame(columns=columns)

        if self.params is None:
            # optimization not done
            params = [None] * len(self.init_guess)
        else:
            params = self.params

        if self.param_ids is None:
            # see self.n_params
            param_ids = range(len(self.ex_ops))
        else:
            param_ids = self.param_ids

        data_list = []

        for i, ex_op in zip(param_ids, self.ex_ops):
            bitstring = get_ex_bitstring(self.n_qubits, self.n_elec, ex_op, self.hcb)
            data_list.append((ex_op, bitstring, params[i], self.init_guess[i]))
        return pd.DataFrame(data_list, columns=columns)

    def print_excitations(self):
        print(self.get_excitation_dataframe().to_string())

    def get_energy_dataframe(self) -> pd.DataFrame:
        """
        Returns energy information dataframe
        """
        if self.params is None:
            series_dict = {"HF": self.e_hf, "MP2": self.e_mp2, "CCSD": self.e_ccsd, "FCI": self.e_fci}
        else:
            ucc_name = self.__class__.__name__
            series_dict = {
                "HF": self.e_hf,
                "MP2": self.e_mp2,
                "CCSD": self.e_ccsd,
                ucc_name: self.energy(),
                "FCI": self.e_fci,
            }
        df = pd.DataFrame()
        energy = pd.Series(series_dict)
        df["energy (Hartree)"] = energy
        if self.e_fci is not None:
            df["error (mH)"] = 1e3 * (energy - self.e_fci)
            df["correlation energy (%)"] = 100 * (energy - self.e_hf) / (self.e_fci - self.e_hf)
        return df

    def print_energy(self):
        df = self.get_energy_dataframe()

        def format_ce(f):
            return f"{f:.3f}"

        formatters = {"correlation energy (%)": format_ce}
        print(df.to_string(index=True, formatters=formatters))

    def print_summary(self, include_circuit: bool = False):
        """
        Print a summary of the class.

        Parameters
        ----------
        include_circuit: bool
            Whether include the circuit section.

        """
        print("################################ Ansatz ###############################")
        self.print_ansatz()
        if self.init_state is not None:
            print("############################ Initial Condition ########################")
            self.print_init_state()
        if include_circuit:
            print("############################### Circuit ###############################")
            self.print_circuit()
        print("############################### Energy ################################")
        self.print_energy()
        print("############################# Excitations #############################")
        self.print_excitations()
        print("######################### Optimization Result #########################")
        if self.opt_res is None:
            print("Optimization not run (.opt_res is None)")
        else:
            print(self.opt_res)

    @property
    def no(self) -> int:
        """The number of occupied orbitals."""
        return self.n_elec // 2

    @property
    def nv(self) -> int:
        """The number of virtual (unoccupied orbitals)."""
        return self.active - self.no

    @property
    def h_fermion_op(self) -> FermionOperator:
        """
        Hamiltonian as openfermion.FermionOperator
        """
        if self.hcb:
            raise ValueError("No FermionOperator available for hard-core boson Hamiltonian")
        return get_hop_from_integral(self.int1e, self.int2e) + self.e_core

    @property
    def h_qubit_op(self) -> QubitOperator:
        """
        Hamiltonian as openfermion.QubitOperator, mapped by
        Jordan-Wigner transformation.
        """
        if not self.hcb:
            return reverse_qop_idx(jordan_wigner(self.h_fermion_op), self.n_qubits)
        else:
            return get_hop_hcb_from_integral(self.int1e, self.int2e)

    @property
    def n_params(self) -> int:
        """The number of parameter in the ansatz/circuit."""
        # this definition ensures that `param[param_id]` is always valid
        if not self.param_ids:
            return 0
        return max(self.param_ids) + 1

    @property
    def statevector_size(self) -> int:
        """The size of the statevector."""
        return 1 << self.n_qubits

    @property
    def civector_size(self) -> int:
        """
        The size of the CI vector.
        """
        if not self.hcb:
            return round(comb(self.n_qubits // 2, self.n_elec // 2)) ** 2
        else:
            return round(comb(self.n_qubits, self.n_elec // 2))

    @property
    def init_state(self) -> Tensor:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self._init_state

    @init_state.setter
    def init_state(self, init_state):
        self._init_state = init_state

    @property
    def params(self) -> Tensor:
        """The circuit parameters."""
        if self._params is not None:
            return self._params
        if self.opt_res is not None:
            return self.opt_res.x
        return None

    @params.setter
    def params(self, params):
        self._params = params

    @property
    def param_ids(self) -> List[int]:
        """The mapping from excitations operators to parameters."""
        if self._param_ids is None:
            if self.ex_ops is None:
                raise ValueError("Excitation operators not defined")
            else:
                return tuple(range(len(self.ex_ops)))
        return self._param_ids

    @param_ids.setter
    def param_ids(self, v):
        self._param_ids = v

    @property
    def param_to_ex_ops(self):
        d = defaultdict(list)
        for i, j in enumerate(self.param_ids):
            d[j].append(self.ex_ops[i])
        return d


def compute_fe_t2(no, nv, int1e, int2e):
    n_orb = no + nv

    def translate_o(n):
        if n % 2 == 0:
            return n // 2 + n_orb
        else:
            return n // 2

    def translate_v(n):
        if n % 2 == 0:
            return n // 2 + no + n_orb
        else:
            return n // 2 + no

    t2 = np.zeros((2 * no, 2 * no, 2 * nv, 2 * nv))
    for i, j, k, l in product(range(2 * no), range(2 * no), range(2 * nv), range(2 * nv)):
        # spin not conserved
        if i % 2 != k % 2 or j % 2 != l % 2:
            continue
        a = translate_o(i)
        b = translate_o(j)
        s = translate_v(l)
        r = translate_v(k)
        if len(set([a, b, s, r])) != 4:
            continue
        # r^ s^ b a
        rr, ss, bb, aa = [i % n_orb for i in [r, s, b, a]]
        if (r < n_orb and s < n_orb) or (r >= n_orb and s >= n_orb):
            e_inter = int2e[aa, rr, bb, ss] - int2e[aa, ss, bb, rr]
        else:
            e_inter = int2e[aa, rr, bb, ss]
        if np.allclose(e_inter, 0):
            continue
        e_diff = _compute_e_diff(r, s, b, a, int1e, int2e, n_orb, no)
        if np.allclose(e_diff, 0):
            raise RuntimeError("RHF degenerate ground state")
        theta = np.arctan(-2 * e_inter / e_diff) / 2
        t2[i, j, k, l] = theta
    return t2


def _compute_e_diff(r, s, b, a, int1e, int2e, n_orb, no):
    inert_a = list(range(no))
    inert_b = list(range(no))
    old_a = []
    old_b = []
    for i in [b, a]:
        if i < n_orb:
            inert_b.remove(i)
            old_b.append(i)
        else:
            inert_a.remove(i % n_orb)
            old_a.append(i % n_orb)

    new_a = []
    new_b = []
    for i in [r, s]:
        if i < n_orb:
            new_b.append(i)
        else:
            new_a.append(i % n_orb)

    diag1e = np.diag(int1e)
    diagj = np.einsum("iijj->ij", int2e)
    diagk = np.einsum("ijji->ij", int2e)

    e_diff_1e = diag1e[new_a].sum() + diag1e[new_b].sum() - diag1e[old_a].sum() - diag1e[old_b].sum()
    # fmt: off
    e_diff_j = _compute_j_outer(diagj, inert_a, inert_b, new_a, new_b) \
               - _compute_j_outer(diagj, inert_a, inert_b, old_a, old_b)
    e_diff_k = _compute_k_outer(diagk, inert_a, inert_b, new_a, new_b) \
               - _compute_k_outer(diagk, inert_a, inert_b, old_a, old_b)
    # fmt: on
    return e_diff_1e + 1 / 2 * (e_diff_j - e_diff_k)


def _compute_j_outer(diagj, inert_a, inert_b, outer_a, outer_b):
    # fmt: off
    v = diagj[inert_a][:, outer_a].sum() + diagj[outer_a][:, inert_a].sum() + diagj[outer_a][:, outer_a].sum() \
      + diagj[inert_a][:, outer_b].sum() + diagj[outer_a][:, inert_b].sum() + diagj[outer_a][:, outer_b].sum() \
      + diagj[inert_b][:, outer_a].sum() + diagj[outer_b][:, inert_a].sum() + diagj[outer_b][:, outer_a].sum() \
      + diagj[inert_b][:, outer_b].sum() + diagj[outer_b][:, inert_b].sum() + diagj[outer_b][:, outer_b].sum()
    # fmt: on
    return v


def _compute_k_outer(diagk, inert_a, inert_b, outer_a, outer_b):
    # fmt: off
    v = diagk[inert_a][:, outer_a].sum() + diagk[outer_a][:, inert_a].sum() + diagk[outer_a][:, outer_a].sum() \
      + diagk[inert_b][:, outer_b].sum() + diagk[outer_b][:, inert_b].sum() + diagk[outer_b][:, outer_b].sum()
    # fmt: on
    return v
