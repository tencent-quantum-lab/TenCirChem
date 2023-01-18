import logging
from functools import partial
from time import time
from typing import Callable, Union, Any, List, Tuple

import numpy as np
from scipy.optimize import minimize
from noisyopt import minimizeSPSA

from openfermion import (
    jordan_wigner,
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
    get_statevector,
    get_densitymatrix,
    get_energy_tensornetwork,
    get_energy_tensornetwork_noise,
    get_energy_tensornetwork_noise_shot,
    get_energy_and_grad_tensornetwork,
    get_energy_and_grad_tensornetwork_noise,
    get_energy_and_grad_tensornetwork_noise_shot,
)
from tencirchem.static.hamiltonian import get_hop_from_integral
from tencirchem.utils.misc import reverse_fop_idx, scipy_opt_wrap
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


def get_init_circuit_parity(n_sorb, n_elec):
    n_qubits = n_sorb - 2
    occupation_list = ([0] * (n_sorb // 2 - n_elec // 2) + [1] * (n_elec // 2)) * 2
    parity_list = (np.cumsum(occupation_list) % 2).tolist()
    parity_list.pop(n_sorb // 2)
    parity_list = parity_list[:-1]
    assert len(parity_list) == n_qubits
    c = Circuit(n_qubits)
    for i in np.nonzero(parity_list)[0]:
        c.X(i)
    return c


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
    for l in range(n_layers):
        for i in range(n_qubits - 1):
            c.cnot(i, (i + 1))
        for i in range(n_qubits):
            c.ry(i, theta=params[l + 1, i])
    return c


class HEA:
    """
    Run hardware-efficient ansatz calculation.
    For a comprehensive tutorial see :doc:`/tutorial_jupyter/noisy_simulation`.
    """

    @classmethod
    def ry(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: int,
        e_core: float,
        n_layers: int,
        init_circuit: Circuit = None,
        **kwargs,
    ):
        """
        Construct the HEA class from electron integrals and custom quantum circuit.
        Overlap integral is assumed to be identity.
        Parity transformation is used to transform from fermion operators to qubit operators.

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
            The initial circuit before the :math:`R_y` ansatz. Defaults to None,
            which creates an HF initial state.

        kwargs:
            Other arguments to be passed to the :func:`__init__` function such as ``engine``.

        Returns
        -------
        hea: :class:`HEA`
             An HEA instance
        """
        n_sorb = 2 * len(int1e)
        n_qubits = n_sorb - 2
        init_guess = np.random.random((n_layers + 1, n_qubits)).ravel()

        def get_circuit(params):
            if init_circuit is None:
                c = Circuit(n_qubits)
            else:
                c = Circuit.from_qir(init_circuit.to_qir())
            return c.append(get_ry_circuit(params, n_qubits, n_layers))

        return cls.from_integral(int1e, int2e, n_elec, e_core, "parity", get_circuit, init_guess, **kwargs)

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
            The fermion to qubit mapping. Supported mappings are ``"parity"`` and ``"jordan-wigner"``.
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
        hop = get_hop_from_integral(int1e, int2e) + e_core
        n_sorb = 2 * len(int1e)
        if mapping == "parity":
            h_qubit_op = parity(hop, n_sorb, n_elec)
        elif mapping == "jordan-wigner":
            h_qubit_op = jordan_wigner(hop)
        else:
            raise ValueError(f"Unknown mapping: {mapping}")

        instance = cls(h_qubit_op, circuit, init_guess, **kwargs)

        instance.int1e = int1e
        instance.int2e = int2e
        instance.e_core = e_core
        instance.hop = hop
        return instance

    def __init__(
        self,
        h_qubit_op: QubitOperator,
        circuit: Union[Callable, QuantumCircuit],
        init_guess: Union[List[float], np.ndarray],
        engine: str = None,
        engine_conf: NoiseConf = None,
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
        if engine.startswith("tensornetwork-noise") and engine_conf is None:
            engine_conf = get_noise_conf()

        init_guess = np.array(init_guess)

        self.h_qubit_op = h_qubit_op
        self.h_array = np.array(get_sparse_operator(self.h_qubit_op).todense())

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

        if init_guess.ndim != 1:
            raise ValueError(f"Init guess should be one-dimensional. Got shape {init_guess}")
        self.init_guess = init_guess
        self.engine = engine
        self.engine_conf = engine_conf
        self.shots = 4096
        self._grad = "param-shift"

        self._params = None
        self.opt_res = None

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
        supported_engine = [None, "tensornetwork", "tensornetwork-noise", "tensornetwork-noise&shot", "qpu"]
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
        Only valid for ``"tensornetwork"`` engine.

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
            assert self.engine == "qpu"
            assert False
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
            Note that ``"autodiff"`` is not compatible with ``"tensornetwork-noise&shot"`` engine.

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
            e, grad = get_energy_and_grad_tensornetwork(params, self.h_array, self.get_circuit, grad)
        elif engine == "tensornetwork-noise":
            e, grad = get_energy_and_grad_tensornetwork_noise(
                params, self.h_array, self.get_dmcircuit_no_noise, self.engine_conf, grad
            )
        elif engine == "tensornetwork-noise&shot":
            if grad == "autodiff":
                raise ValueError(f"Engine {engine} is incompatible with grad method {grad}")
            e, grad = get_energy_and_grad_tensornetwork_noise_shot(
                params,
                tuple(self.h_qubit_op.terms.keys()),
                list(self.h_qubit_op.terms.values()),
                self.get_dmcircuit_no_noise,
                self.engine_conf,
                self.shots,
                grad,
            )
        else:
            assert self.engine == "qpu"
            assert False
        return e, grad

    def kernel(self):
        logger.info("Begin optimization")

        func, stating_time = self.get_opt_function(with_time=True)
        time1 = time()
        if self.grad == "free":
            if self.engine in ["tensornetwork", "tensornetwork-noise"]:
                opt_res = minimize(func, x0=self.init_guess, method="COBYLA")
            else:
                assert self.engine == "tensornetwork-noise&shot"
                opt_res = minimizeSPSA(func, x0=self.init_guess, paired=False, niter=125)
        else:
            opt_res = minimize(func, x0=self.init_guess, jac=True, method="L-BFGS-B")

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
