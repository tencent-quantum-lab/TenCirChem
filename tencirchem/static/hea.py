#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import logging
from functools import partial
from itertools import product
from time import time
from typing import Callable, Union, Any, List, Tuple

import numpy as np
from scipy.optimize import minimize
from noisyopt import minimizeSPSA
from pyscf.gto.mole import Mole
from openfermion import (
    hermitian_conjugated,
    jordan_wigner,
    bravyi_kitaev,
    binary_code_transform,
    checksum_code,
    parity_code,
    get_sparse_operator,
    QubitOperator,
    FermionOperator,
)
from qiskit import QuantumCircuit
import tensorcircuit as tc
from tensorcircuit import Circuit, DMCircuit
from tensorcircuit.noisemodel import NoiseConf, circuit_with_noise

from tencirchem.static.engine_hea import (
    QpuConf,
    get_statevector,
    get_densitymatrix,
    get_energy_tensornetwork,
    get_energy_tensornetwork_noise,
    get_energy_tensornetwork_shot,
    get_energy_tensornetwork_noise_shot,
    get_energy_qpu,
    get_energy_and_grad_tensornetwork,
    get_energy_and_grad_tensornetwork_noise,
    get_energy_and_grad_tensornetwork_shot,
    get_energy_and_grad_tensornetwork_noise_shot,
    get_energy_and_grad_qpu,
)
from tencirchem.static.hamiltonian import get_hop_from_integral
from tencirchem.utils.misc import reverse_fop_idx, scipy_opt_wrap, reverse_qop_idx
from tencirchem.utils.circuit import get_circuit_dataframe

logger = logging.getLogger(__file__)


Tensor = Any


def get_noise_conf(probability=0.02):
    noise_conf = NoiseConf()
    # note TC definition of error probability
    channel = tc.channels.isotropicdepolarizingchannel(probability, 2)

    def condition(qir):
        return len(qir["index"]) == 2

    noise_conf.add_noise_by_condition(condition, channel)
    return noise_conf


def binary(fermion_operator: FermionOperator, n_modes: int, n_elec: int) -> QubitOperator:
    """
    Performs binary transformation.

    Parameters
    ----------
    fermion_operator: FermionOperator
        The fermion operator.
    n_modes: int
        The number of modes.
    n_elec: int
        The number of electrons.

    Returns
    -------
    qubit_operator: QubitOperator
    """
    return binary_code_transform(fermion_operator, 2 * checksum_code(n_modes // 2, (n_elec // 2) % 2))


def _parity(fermion_operator, n_modes):
    return binary_code_transform(fermion_operator, parity_code(n_modes))


def parity(fermion_operator: FermionOperator, n_modes: int, n_elec: int) -> QubitOperator:
    """
    Performs parity transformation.

    Parameters
    ----------
    fermion_operator: FermionOperator
        The fermion operator.
    n_modes: int
        The number of modes (spin-orbitals).
    n_elec: int
        The number of electrons.

    Returns
    -------
    qubit_operator: QubitOperator
    """
    qubit_operator = _parity(reverse_fop_idx(fermion_operator, n_modes), n_modes)
    res = 0
    assert n_modes % 2 == 0
    reduction_indices = [n_modes // 2 - 1, n_modes - 1]
    phase_alpha = (-1) ** (n_elec // 2)
    for qop in qubit_operator:
        # qop example: 0.5 [Z1 X2 X3]
        pauli_string, coeff = next(iter(qop.terms.items()))
        # pauli_string example: ((1, 'Z'), (2, 'X'), (3, 'X'))
        # coeff example: 0.5
        new_pauli_string = []
        for idx, symbol in pauli_string:
            is_alpha = idx <= reduction_indices[0]
            if idx in reduction_indices:
                if symbol in ["X", "Y"]:
                    # discard this term because the bit will never change
                    continue
                else:
                    assert symbol == "Z"
                    if is_alpha:
                        coeff *= phase_alpha
                    continue
            if not is_alpha:
                idx -= 1
            new_pauli_string.append((idx, symbol))
        qop.terms = {tuple(new_pauli_string): coeff}
        res += qop
    return res


def fop_to_qop(fop: FermionOperator, mapping: str, n_sorb: int, n_elec: int) -> QubitOperator:
    if mapping == "parity":
        qop = parity(fop, n_sorb, n_elec)
    elif mapping in ["jordan-wigner", "jordan_wigner"]:
        qop = reverse_qop_idx(jordan_wigner(fop), n_sorb)
    elif mapping in ["bravyi-kitaev", "bravyi_kitaev"]:
        qop = reverse_qop_idx(bravyi_kitaev(fop, n_sorb), n_sorb)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")
    return qop


def get_ry_circuit(params: Tensor, n_qubits: int, n_layers: int) -> Circuit:
    """
    Get the parameterized :math:`R_y` circuit.

    Parameters
    ----------
    params: Tensor
        The circuit parameters.
    n_qubits: int
        The number of qubits.
    n_layers: int
        The number of layers in the ansatz.

    Returns
    -------
    c: Circuit
    """
    c = Circuit(n_qubits)
    params = params.reshape(n_layers + 1, n_qubits)
    for i in range(n_qubits):
        c.ry(i, theta=params[0, i])
    c.barrier_instruction(*range(n_qubits))
    for l in range(n_layers):
        for i in range(n_qubits - 1):
            c.cnot(i, (i + 1))
        for i in range(n_qubits):
            c.ry(i, theta=params[l + 1, i])
        c.barrier_instruction(*range(n_qubits))
    return c


class HEA:
    """
    Run hardware-efficient ansatz calculation.
    For a comprehensive tutorial see :doc:`/tutorial_jupyter/noisy_simulation`.
    """

    @classmethod
    def from_molecule(cls, m: Mole, active_space=None, n_layers=3, mapping="parity", **kwargs):
        """
        Construct the HEA class from the given molecule.
        The :math:`R_y` ansatz is employed. By default, the number of layers is set to 3.


        Parameters
        ----------
        m: Mole
            The molecule object.
        active_space: Tuple[int, int], optional
            Active space approximation. The first integer is the number of electrons and the second integer is
            the number or spatial-orbitals. Defaults to None.
        n_layers: int
            The number of layers in the :math:`R_y` ansatz. Defaults to 3.
        mapping: str
            The fermion to qubit mapping. Supported mappings are ``"parity"``,
            and ``"bravyi-kitaev"``.

        kwargs:
            Other arguments to be passed to the :func:`__init__` function such as ``engine``.

        Returns
        -------
        hea: :class:`HEA`
             An HEA instance
        """
        from tencirchem import UCC

        ucc = UCC(m, active_space=active_space, run_ccsd=False, run_fci=False)
        return cls.ry(ucc.int1e, ucc.int2e, ucc.n_elec, ucc.e_core, n_layers=n_layers, mapping=mapping, **kwargs)

    @classmethod
    def ry(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: int,
        e_core: float,
        n_layers: int,
        init_circuit: Circuit = None,
        mapping: str = "parity",
        **kwargs,
    ):
        r"""
        Construct the HEA class from electron integrals and the :math:`R_y` ansatz.
        The circuit consists of interleaved layers of $R_y$ and CNOT gates

        .. math::

            |\Psi(\theta)\rangle=\prod_{l=k}^1\left [ L_{R_y}^{(l)}(\theta) L_{CNOT}^{(l)} \right ] L_{R_y}^{(0)}(\theta) |{\phi}\rangle

        where $k$ is the total number of layers, and the layers are defined as

        .. math::
            L_{CNOT}^{(l)}=\prod_{j=N/2-1}^1 CNOT_{2j, 2j+1} \prod_{j=N/2}^{1} CNOT_{2j-1, 2j}

        .. math::
            L_{R_y}^{(l)}(\theta)=\prod_{j=N}^{1} RY_{j}(\theta_{lj})

        Overlap integral is assumed to be identity.
        Parity transformation is used to transform from fermion operators to qubit operators by default.

        Parameters
        ----------
        int1e: np.ndarray
            One-body integral in spatial orbital.
        int2e: np.ndarray
            Two-body integral, in spatial orbital, chemists' notation, and without considering symmetry.
        n_elec: int
            The number of electrons
        e_core: float
            The nuclear energy or core energy if active space approximation is involved.
        n_layers: int
            The number of layers in the ansatz.
        init_circuit: Circuit
            The initial circuit before the :math:`R_y` ansatz. Defaults to None.
        mapping: str
            The fermion to qubit mapping. Supported mappings are ``"parity"``,
            and ``"bravyi-kitaev"``.

        kwargs:
            Other arguments to be passed to the :func:`__init__` function such as ``engine``.

        Returns
        -------
        hea: :class:`HEA`
             An HEA instance
        """
        n_sorb = 2 * len(int1e)
        if mapping == "parity":
            n_qubits = n_sorb - 2
        elif mapping in ["jordan-wigner", "jordan_wigner", "bravyi-kitaev", "bravyi_kitaev"]:
            n_qubits = n_sorb
        else:
            raise ValueError(f"Unknown mapping: {mapping}")

        init_guess = np.random.random((n_layers + 1, n_qubits)).ravel()

        def get_circuit(params):
            if init_circuit is None:
                c = Circuit(n_qubits)
            else:
                c = Circuit.from_qir(init_circuit.to_qir(), init_circuit.circuit_param)
            return c.append(get_ry_circuit(params, n_qubits, n_layers))

        instance = cls.from_integral(int1e, int2e, n_elec, e_core, mapping, get_circuit, init_guess, **kwargs)
        instance.mapping = mapping
        return instance

    @classmethod
    def from_integral(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: int,
        e_core: float,
        mapping: str,
        circuit: Union[Callable, QuantumCircuit],
        init_guess: np.ndarray,
        **kwargs,
    ):
        """
        Construct the HEA class from electron integrals and custom quantum circuit.
        Overlap integral is assumed to be identity.

        Parameters
        ----------
        int1e: np.ndarray
            One-body integral in spatial orbital.
        int2e: np.ndarray
            Two-body integral, in spatial orbital, chemists' notation, and without considering symmetry.
        n_elec: int
            The number of electrons
        e_core: float
            The nuclear energy or core energy if active space approximation is involved.
        mapping: str
            The fermion to qubit mapping. Supported mappings are ``"parity"``,
            and ``"bravyi-kitaev"``.
        circuit: Callable or QuantumCircuit
            The ansatz as a function or Qiskit :class:`QuantumCircuit`
        init_guess: list of float or :class:`np.ndarray`
            The parameter initial guess.

        kwargs:
            Other arguments to be passed to the :func:`__init__` function such as ``engine``.

        Returns
        -------
        hea: :class:`HEA`
             An HEA instance
        """

        if isinstance(n_elec, tuple):
            if len(n_elec) != 2 or n_elec[0] != n_elec[1]:
                raise ValueError(f"Incompatible n_elec: {n_elec}")
            n_elec = n_elec[0] + n_elec[1]

        hop = get_hop_from_integral(int1e, int2e) + e_core
        n_sorb = 2 * len(int1e)
        h_qubit_op = fop_to_qop(hop, mapping, n_sorb, n_elec)

        instance = cls(h_qubit_op, circuit, init_guess, **kwargs)

        instance.mapping = mapping
        instance.int1e = int1e
        instance.int2e = int2e
        instance.n_elec = n_elec
        instance.e_core = e_core
        instance.hop = hop
        return instance

    @classmethod
    def as_pyscf_solver(cls, config_function: Callable = None, opt_engine: str = None, **kwargs):
        """
        Converts the ``HEA`` class to a PySCF FCI solver using :math:`R_y` ansatz.

        Parameters
        ----------
        config_function: callable
            A function to configure the ``HEA`` instance.
            Accepts the ``HEA`` instance and modifies it inplace before :func:`kernel` is called.
        opt_engine: str
            The engine to use when performing the circuit parameter optimization.
        kwargs
            Other arguments to be passed to the :func:`__init__` function such as ``engine``.

        Returns
        -------
        FCISolver

        Examples
        --------
        >>> from pyscf.mcscf import CASSCF
        >>> from tencirchem import HEA
        >>> from tencirchem.molecule import h8
        >>> # normal PySCF workflow
        >>> hf = h8.HF()
        >>> round(hf.kernel(), 6)
        -4.149619
        >>> casscf = CASSCF(hf, 2, 2)
        >>> # set the FCI solver for CASSCF to be HEA
        >>> casscf.fcisolver = HEA.as_pyscf_solver(n_layers=1)
        >>> round(casscf.kernel()[0], 6)
        -4.166473
        """

        class FakeFCISolver:
            def __init__(self):
                self.instance: HEA = None
                self.config_function = config_function
                self.instance_kwargs = kwargs.copy()
                if "n_layers" not in self.instance_kwargs:
                    self.instance_kwargs["n_layers"] = 1

            def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
                self.instance = cls.ry(h1, h2, nelec, e_core=ecore, **self.instance_kwargs)
                if self.config_function is not None:
                    self.config_function(self.instance)
                if opt_engine is None:
                    e = self.instance.kernel()
                else:
                    engine_bak = self.instance.engine
                    self.instance.engine = opt_engine
                    self.instance.kernel()
                    self.instance.engine = engine_bak
                    e = self.instance.energy()
                return e, self.instance.params

            def make_rdm1(self, params, norb, nelec):
                rdm1 = self.instance.make_rdm1(params)
                return rdm1

            def make_rdm12(self, params, norb, nelec):
                rdm1 = self.instance.make_rdm1(params)
                rdm2 = self.instance.make_rdm2(params)
                return rdm1, rdm2

            def spin_square(self, params, norb, nelec):
                return 0, 1

        return FakeFCISolver()

    def __init__(
        self,
        h_qubit_op: QubitOperator,
        circuit: Union[Callable, QuantumCircuit],
        init_guess: Union[List[float], np.ndarray],
        engine: str = None,
        engine_conf: [NoiseConf, QpuConf] = None,
    ):
        """
        Construct the HEA class from Hamiltonian in :class:`QubitOperator` form and the ansatz.

        Parameters
        ----------
        h_qubit_op: QubitOperator
            Hamiltonian as openfermion :class:`QubitOperator`.
        circuit: Callable or QuantumCircuit
            The ansatz as a function or Qiskit :class:`QuantumCircuit`
        init_guess: list of float or :class:`np.ndarray`
            The parameter initial guess.
        engine: str, optional
            The engine to run the calculation. See :ref:`advanced:Engines` for details.
            Defaults to ``"tensornetwork"``.
        engine_conf: NoiseConf, optional
            The noise configuration for the circuit. Defaults to None, in which case
            if a noisy engine is used, an isotropic depolarizing error channel for all 2-qubit gates
            with :math:`p=0.02` is added to the circuit.
        """
        self._check_engine(engine)

        if engine is None:
            engine = "tensornetwork"
        if engine == "tensornetwork" and engine_conf is not None:
            raise ValueError("Tensornetwork engine does not have engine configuration")
        if engine_conf is None:
            if engine.startswith("tensornetwork-noise"):
                engine_conf = get_noise_conf()
            if engine.startswith("qpu"):
                engine_conf = QpuConf()

        init_guess = np.array(init_guess)

        self.h_qubit_op = h_qubit_op

        if isinstance(circuit, Callable):
            self.get_circuit = circuit
            # sanity check
            c: Circuit = self.get_circuit(init_guess)
            if isinstance(c, DMCircuit):
                raise TypeError("`circuit` function should return Circuit instead of DMCircuit")
            self.n_qubits = c.circuit_param["nqubits"]
        elif isinstance(circuit, QuantumCircuit):

            def get_circuit(params):
                return Circuit.from_qiskit(circuit, binding_params=params)

            self.get_circuit = get_circuit
            assert circuit.num_parameters == len(init_guess)
            self.n_qubits = circuit.num_qubits
        else:
            raise TypeError("circuit must be callable or qiskit QuantumCircuit")

        self.h_array = np.array(get_sparse_operator(self.h_qubit_op, self.n_qubits).todense())

        if init_guess.ndim != 1:
            raise ValueError(f"Init guess should be one-dimensional. Got shape {init_guess}")
        self.init_guess = init_guess
        self.engine = engine
        self.engine_conf = engine_conf
        self.shots = 4096
        self._grad = "param-shift"

        self.scipy_minimize_options = None
        self._params = None
        self.opt_res = None

        # allow setting these attributes for features such as calculating RDM
        # could make it a function for customized mapping
        self.mapping: str = None  # fermion-to-qubit mapping
        self.int1e = None
        self.int2e = None
        self.n_elec = None
        self.e_core = None
        self.hop = None

    def get_dmcircuit(self, params: Tensor = None, noise_conf: NoiseConf = None) -> DMCircuit:
        """
        Get the :class:`DMCircuit` with noise.
        Only valid for ``"tensornetwork-noise"`` and ``"tensornetwork-noise&shot"`` engines.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        noise_conf: NoiseConf, optional
            The noise configuration for the circuit. Defaults to None, in which case
            ``self.engine_conf`` is used.

        Returns
        -------
        dmcircuit: DMCircuit
        """
        params = self._check_params_argument(params)
        dmcircuit = self.get_dmcircuit_no_noise(params)
        if noise_conf is None:
            assert self.engine.startswith("tensornetwork-noise")
            noise_conf = self.engine_conf
        dmcircuit = circuit_with_noise(dmcircuit, noise_conf)
        return dmcircuit

    def get_dmcircuit_no_noise(self, params: Tensor = None) -> DMCircuit:
        """
        Get the :class:`DMCircuit` without noise.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.

        Returns
        -------
        dmcircuit: DMCircuit
        """
        qir = self.get_circuit(params).to_qir()
        dmcircuit = DMCircuit.from_qir(qir)
        return dmcircuit

    def _check_engine(self, engine):
        supported_engine = [
            None,
            "tensornetwork",
            "tensornetwork-noise",
            "tensornetwork-shot",
            "tensornetwork-noise&shot",
            "qpu",
        ]
        if not engine in supported_engine:
            raise ValueError(f"Engine '{engine}' not supported")

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

    def statevector(self, params: Tensor = None) -> Tensor:
        """
        Evaluate the circuit state vector without considering noise.
        Valid for ``"tensornetwork"`` and ``"tensornetwork-shot"`` engine.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.

        Returns
        -------
        statevector: Tensor
            Corresponding state vector

        See Also
        --------
        densitymatrix: Evaluate the circuit density matrix in the presence of circuit noise.
        energy: Evaluate the total energy.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> import numpy as np
        >>> from tencirchem import UCC, HEA
        >>> from tencirchem.molecule import h2
        >>> ucc = UCC(h2)
        >>> hea = HEA.ry(ucc.int1e, ucc.int2e, ucc.n_elec, ucc.e_core, n_layers=1)
        >>> np.round(hea.statevector([0, np.pi, 0, 0]), 8)  # HF state
        array([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j])
        """
        # Evaluate the circuit state vector without noise.
        # only engine is "tensornetwork"
        params = self._check_params_argument(params)
        statevector = get_statevector(params, self.get_circuit)
        return statevector

    def densitymatrix(self, params: Tensor = None) -> Tensor:
        """
        Evaluate the circuit density matrix in the presence of circuit noise.
        Only valid for ``"tensornetwork-noise"`` and ``"tensornetwork-noise&shot"`` engines.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.

        Returns
        -------
        densitymatrix: Tensor

        See Also
        --------
        statevector: Evaluate the circuit state vector.
        energy: Evaluate the total energy.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> import numpy as np
        >>> from tencirchem import UCC, HEA
        >>> from tencirchem.molecule import h2
        >>> ucc = UCC(h2)
        >>> hea = HEA.ry(ucc.int1e, ucc.int2e, ucc.n_elec, ucc.e_core,
        ...         n_layers=1, engine="tensornetwork-noise")
        >>> np.round(hea.densitymatrix([0, np.pi, 0, 0]), 8)  # HF state with noise
        array([[0.00533333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j],
               [0.        +0.j, 0.984     +0.j, 0.        +0.j, 0.        +0.j],
               [0.        +0.j, 0.        +0.j, 0.00533333+0.j, 0.        +0.j],
               [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.00533333+0.j]])
        """
        # engines are "tensornetwork-noise" and "tensornetwork-noise&shot"
        params = self._check_params_argument(params)
        # the last two arguments should be identical and not garbage collected for each call for `jit` to work
        densitymatrix = get_densitymatrix(params, self.get_dmcircuit_no_noise, self.engine_conf)
        return densitymatrix

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
        statevector: Evaluate the circuit state vector.
        densitymatrix: Evaluate the circuit density matrix in the presence of circuit noise.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> import numpy as np
        >>> from tencirchem import UCC, HEA
        >>> from tencirchem.molecule import h2
        >>> ucc = UCC(h2)
        >>> hea = HEA.ry(ucc.int1e, ucc.int2e, ucc.n_elec, ucc.e_core,
        ...         n_layers=1, engine="tensornetwork-noise")
        >>> # HF state, no noise
        >>> round(hea.energy([0, np.pi, 0, 0], "tensornetwork"), 8)
        -1.11670614
        >>> # HF state, gate noise
        >>> round(hea.energy([0, np.pi, 0, 0], "tensornetwork-noise"), 4)
        -1.1001
        >>> # HF state, measurement noise. Set the number of shots by `hea.shots`
        >>> round(hea.energy([0, np.pi, 0, 0], "tensornetwork-shot"), 1)
        -1.1
        >>> # HF state, gate+measurement noise
        >>> hea.energy([0, np.pi, 0, 0], "tensornetwork-noise&shot")  # doctest:+ELLIPSIS
        -1...
        """
        params = self._check_params_argument(params)
        if engine is None:
            engine = self.engine
        if engine == "tensornetwork":
            e = get_energy_tensornetwork(params, self.h_array, self.get_circuit)
        elif engine == "tensornetwork-noise":
            e = get_energy_tensornetwork_noise(params, self.h_array, self.get_dmcircuit_no_noise, self.engine_conf)
        elif engine == "tensornetwork-shot":
            e = get_energy_tensornetwork_shot(
                params,
                tuple(self.h_qubit_op.terms.keys()),
                list(self.h_qubit_op.terms.values()),
                self.get_circuit,
                self.shots,
            )
        elif engine == "tensornetwork-noise&shot":
            e = get_energy_tensornetwork_noise_shot(
                params,
                tuple(self.h_qubit_op.terms.keys()),
                list(self.h_qubit_op.terms.values()),
                self.get_dmcircuit_no_noise,
                self.engine_conf,
                self.shots,
            )
        else:
            assert engine == "qpu"
            e = get_energy_qpu(
                params,
                tuple(self.h_qubit_op.terms.keys()),
                list(self.h_qubit_op.terms.values()),
                self.get_circuit,
                self.engine_conf,
                self.shots,
            )
        return e

    def energy_and_grad(self, params: Tensor = None, engine: str = None, grad: str = None) -> Tuple[float, Tensor]:
        """
        Evaluate the total energy and parameter gradients using parameter-shift rule.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.
        grad: str, optional
            The algorithm to use for the gradient. Defaults to ``None``, which means ``self.grad`` will be used.
            Possible options are ``"param-shift"`` for parameter-shift rule and
            ``"autodiff"`` for auto-differentiation.
            Note that ``"autodiff"`` is not compatible with ``"tensornetwork-shot"``
            and ``"tensornetwork-noise&shot"`` engine.

        Returns
        -------
        energy: float
            Total energy
        grad: Tensor
            The parameter gradients

        See Also
        --------
        statevector: Evaluate the circuit state vector.
        densitymatrix: Evaluate the circuit density matrix in the presence of circuit noise.
        energy: Evaluate the total energy.
        """

        params = self._check_params_argument(params)

        if engine is None:
            engine = self.engine

        if grad is None:
            grad = self.grad

        if grad == "free":
            raise ValueError("Must provide a gradient algorithm")

        if engine == "tensornetwork":
            e, grad_array = get_energy_and_grad_tensornetwork(params, self.h_array, self.get_circuit, grad)
        elif engine == "tensornetwork-noise":
            e, grad_array = get_energy_and_grad_tensornetwork_noise(
                params, self.h_array, self.get_dmcircuit_no_noise, self.engine_conf, grad
            )
        elif engine == "tensornetwork-shot":
            if grad == "autodiff":
                raise ValueError(f"Engine {engine} is incompatible with grad method {grad}")
            e, grad_array = get_energy_and_grad_tensornetwork_shot(
                params,
                tuple(self.h_qubit_op.terms.keys()),
                list(self.h_qubit_op.terms.values()),
                self.get_circuit,
                self.shots,
                grad,
            )
        elif engine == "tensornetwork-noise&shot":
            if grad == "autodiff":
                raise ValueError(f"Engine {engine} is incompatible with grad method {grad}")
            e, grad_array = get_energy_and_grad_tensornetwork_noise_shot(
                params,
                tuple(self.h_qubit_op.terms.keys()),
                list(self.h_qubit_op.terms.values()),
                self.get_dmcircuit_no_noise,
                self.engine_conf,
                self.shots,
                grad,
            )
        else:
            assert engine == "qpu"
            if grad == "autodiff":
                raise ValueError(f"Engine {engine} is incompatible with grad method {grad}")
            e, grad_array = get_energy_and_grad_qpu(
                params,
                tuple(self.h_qubit_op.terms.keys()),
                list(self.h_qubit_op.terms.values()),
                self.get_circuit,
                self.shots,
                grad,
            )
        return e, grad_array

    def kernel(self):
        logger.info("Begin optimization")

        func, stating_time = self.get_opt_function(with_time=True)

        time1 = time()
        if self.grad == "free":
            if self.engine in ["tensornetwork", "tensornetwork-noise", "qpu"]:
                opt_res = minimize(func, x0=self.init_guess, method="COBYLA", options=self.scipy_minimize_options)
            else:
                assert self.engine in ["tensornetwork-shot", "tensornetwork-noise&shot"]
                opt_res = minimizeSPSA(func, x0=self.init_guess, paired=False, niter=100, disp=True)
        else:
            opt_res = minimize(
                func, x0=self.init_guess, jac=True, method="L-BFGS-B", options=self.scipy_minimize_options
            )

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

    def get_opt_function(self, grad: str = None, with_time: bool = False) -> Union[Callable, Tuple[Callable, float]]:
        """
        Returns the cost function in SciPy format for optimization.
        Basically a wrapper to :func:`energy_and_grad` or :func:`energy`,


        Parameters
        ----------
        with_time: bool, optional
            Whether return staging time. Defaults to False.
        grad: str, optional
            The algorithm to use for the gradient. Defaults to ``None``, which means ``self.grad`` will be used.
            Possible options are ``"param-shift"`` for parameter-shift rule and
            ``"autodiff"`` for auto-differentiation.
            Note that ``"autodiff"`` is not compatible with ``"tensornetwork-noise&shot"`` engine.
        Returns
        -------
        opt_function: Callable
            The optimization cost function in SciPy format.
        time: float
            Staging time. Returned when ``with_time`` is set to ``True``.
        """
        if grad is None:
            grad = self.grad

        if grad != "free":
            func = scipy_opt_wrap(partial(self.energy_and_grad, engine=self.engine))
        else:
            func = scipy_opt_wrap(partial(self.energy, engine=self.engine), gradient=False)

        time1 = time()
        if tc.backend.name == "jax":
            logger.info("JIT compiling the circuit")
            _ = func(np.zeros(self.n_params))
            logger.info("Circuit JIT compiled")
        time2 = time()
        if with_time:
            return func, time2 - time1
        return func

    def make_rdm1(self, params: Tensor = None) -> np.ndarray:
        r"""
        Evaluate the spin-traced one-body reduced density matrix (1RDM).

        .. math::

            \textrm{1RDM}[p,q] = \langle p_{\alpha}^\dagger q_{\alpha} \rangle
                + \langle p_{\beta}^\dagger q_{\beta} \rangle

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.

        Returns
        -------
        rdm1: np.ndarray
            The spin-traced one-body RDM.

        See Also
        --------
        make_rdm2: Evaluate the spin-traced two-body reduced density matrix (2RDM).
        """

        if params is None:
            params = self._check_params_argument(params)
        if self.mapping is None:
            raise ValueError("Must first set the fermion-to-qubit mapping")

        if self.mapping == "parity":
            n_sorb = self.n_qubits + 2
        else:
            n_sorb = self.n_qubits
        n_orb = n_sorb // 2
        rdm1 = np.zeros([n_orb] * 2)

        # assuming closed shell
        # could optimize for tn engine by caching the statevector or dm
        for i in range(n_orb):
            for j in range(i + 1):
                fop = FermionOperator(f"{i}^ {j}")
                fop = fop + hermitian_conjugated(fop)
                qop = fop_to_qop(fop, self.mapping, n_sorb, self.n_elec)
                hea = HEA(qop, self.get_circuit, params, self.engine, self.engine_conf)
                # for spin orbital RDM
                v = hea.energy(params) / 2
                # spatial orbital RDM
                rdm1[i, j] = rdm1[j, i] = 2 * v

        return rdm1

    def make_rdm2(self, params: Tensor = None) -> np.ndarray:
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


        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.

        Returns
        -------
        rdm2: np.ndarray
            The spin-traced two-body RDM.

        See Also
        --------
        make_rdm1: Evaluate the spin-traced one-body reduced density matrix (1RDM).
        """
        if params is None:
            params = self._check_params_argument(params)
        if self.mapping is None:
            raise ValueError("Must first set the fermion-to-qubit mapping")

        if self.mapping == "parity":
            n_sorb = self.n_qubits + 2
        else:
            n_sorb = self.n_qubits
        n_orb = n_sorb // 2
        rdm2 = np.zeros([n_orb] * 4)

        calculated_indices = set()
        # a^\dagger_p a^\dagger_q a_r a_s
        # possible spins: aaaa, abba, baab, bbbb
        for p, q, r, s in product(range(n_orb), repeat=4):
            if (p, q, r, s) in calculated_indices:
                continue
            # aaaa is the same as bbbb, abba is the same as baab
            fop_aaaa = FermionOperator(f"{p}^ {q}^ {r} {s}")
            fop_abba = FermionOperator(f"{p}^ {q+n_orb}^ {r+n_orb} {s}")
            fop = fop_aaaa + fop_abba
            fop = fop + hermitian_conjugated(fop)
            qop = fop_to_qop(fop, self.mapping, n_sorb, self.n_elec)
            hea = HEA(qop, self.get_circuit, params, self.engine, self.engine_conf)
            # for spin RDM
            v = hea.energy(params) / 2
            indices = [(p, q, r, s), (s, r, q, p), (q, p, s, r), (r, s, p, q)]
            for idx in indices:
                # 2* for spatial RDM
                rdm2[idx] = 2 * v
                calculated_indices.add(idx)
        # transpose to PySCF notation: rdm2[p,q,r,s] = <p^+ r^+ s q>
        rdm2 = rdm2.transpose(0, 3, 1, 2)
        return rdm2

    def print_circuit(self):
        c = self.get_circuit(self.init_guess)
        df = get_circuit_dataframe(c)

        def format_flop(f):
            return f"{f:.3e}"

        formatters = {"flop": format_flop}
        print(df.to_string(index=True, formatters=formatters))

    def print_summary(self):
        print("############################### Circuit ###############################")
        self.print_circuit()
        print("######################### Optimization Result #########################")
        if self.opt_res is None:
            print("Optimization not run yet")
        else:
            print(self.opt_res)

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, v):
        if not v in ["param-shift", "autodiff", "free"]:
            raise ValueError(f"Invalid gradient method {v}")
        self._grad = v

    @property
    def n_params(self):
        """The number of parameter in the ansatz/circuit."""
        return len(self.init_guess)

    @property
    def params(self):
        """The circuit parameters."""
        if self._params is not None:
            return self._params
        if self.opt_res is not None:
            return self.opt_res.x
        return None

    @params.setter
    def params(self, params):
        self._params = params
