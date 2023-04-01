#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from time import time
import logging
from typing import Tuple, List

import numpy as np
from pyscf.gto.mole import Mole

from tencirchem.static.ucc import UCC

logger = logging.getLogger(__name__)


class KUPCCGSD(UCC):
    """
    Run :math:`k`-UpCCGSD calculation.
    The interfaces are similar to :class:`UCCSD <tencirchem.UCCSD>`.
    """

    def __init__(
        self,
        mol: Mole,
        active_space: Tuple[int, int] = None,
        mo_coeff: np.ndarray = None,
        k: int = 3,
        n_tries: int = 1,
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
        active_space: Tuple[int, int], optional
            Active space approximation. The first integer is the number of electrons and the second integer is
            the number or spatial-orbitals. Defaults to None.
        mo_coeff: np.ndarray, optional
            Molecule coefficients. If provided then RHF is skipped.
            Can be used in combination with the ``init_state`` attribute.
            Defaults to None which means RHF orbitals are used.
        k: int, optional
            The number of layers in the ansatz. Defaults to 3
        n_tries: int, optional
            The number of different initial points used for VQE calculation.
            For large circuits usually a lot of runs are required for good accuracy.
            Defaults to 1.
        engine: str, optional
            The engine to run the calculation. See :ref:`advanced:Engines` for details.
        run_hf: bool, optional
            Whether run HF for molecule orbitals. Defaults to ``True``.
        run_mp2: bool, optional
            Whether run MP2 for energy reference. Defaults to ``True``.
        run_ccsd: bool, optional
            Whether run CCSD for energy reference. Defaults to ``True``.
        run_fci: bool, optional
            Whether run FCI for energy reference. Defaults to ``True``.

        See Also
        --------
        tencirchem.UCCSD
        tencirchem.PUCCD
        tencirchem.UCC
        """
        super().__init__(
            mol,
            init_method=None,
            active_space=active_space,
            mo_coeff=mo_coeff,
            engine=engine,
            run_hf=run_hf,
            run_mp2=run_mp2,
            run_ccsd=run_ccsd,
            run_fci=run_fci,
        )
        # the number of layers
        self.k = k
        # the number of different initialization
        self.n_tries = n_tries
        self.ex_ops, self.param_ids, self.init_guess = self.get_ex_ops(self.t1, self.t2)
        self.init_guess_list = [self.init_guess]
        for _ in range(self.n_tries - 1):
            self.init_guess_list.append(np.random.rand(self.n_params) - 0.5)
        self.e_tries_list = []
        self.opt_res_list = []
        self.staging_time = self.opt_time = None

    def kernel(self):
        _, stating_time = self.get_opt_function(with_time=True)

        time1 = time()
        for i in range(self.n_tries):
            logger.info(f"k-UpCCGSD try {i}")
            if self.n_tries == 1:
                if not np.allclose(self.init_guess, self.init_guess_list[0]):
                    logger.info("Inconsistent `self.init_guess` and `self.init_guess_list`.  Use `self.init_guess`.")
            else:
                self.init_guess = self.init_guess_list[i]
            super().kernel()
            logger.info(f"k-UpCCGSD try {i} energy {self.opt_res.fun}")
            self.opt_res_list.append(self.opt_res)
        self.opt_res_list.sort(key=lambda x: x.fun)
        self.e_tries_list = [float(res.fun) for res in self.opt_res_list]
        time2 = time()

        self.staging_time = stating_time
        self.opt_time = time2 - time1
        self.opt_res = self.opt_res_list[0]
        self.opt_res.e_tries = self.e_tries_list

        if not self.opt_res.success:
            logger.warning("Optimization failed. See `.opt_res` for details.")

        self.init_guess = self.opt_res.init_guess
        return float(self.opt_res.e)

    def get_ex_ops(self, t1: np.ndarray = None, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], np.ndarray]:
        """
        Get one-body and two-body excitation operators for :math:`k`-UpCCGSD ansatz.
        The excitations are generalized and two-body excitations are restricted to paired ones.
        Initial guesses are generated randomly.

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
        init_guess: np.ndarray
            The initial guess for the parameters.

        See Also
        --------
        get_ex1_ops: Get generalized one-body excitation operators.
        get_ex2_ops: Get generalized paired two-body excitation operators.

        Examples
        --------
        >>> from tencirchem import KUPCCGSD
        >>> from tencirchem.molecule import h2
        >>> kupccgsd = KUPCCGSD(h2)
        >>> ex_op, param_ids, init_guess = kupccgsd.get_ex_ops()
        >>> ex_op
        [(1, 3, 2, 0), (3, 2), (1, 0), (1, 3, 2, 0), (3, 2), (1, 0), (1, 3, 2, 0), (3, 2), (1, 0)]
        >>> param_ids
        [0, 1, 1, 2, 3, 3, 4, 5, 5]
        >>> init_guess  # doctest:+ELLIPSIS
        array([...])
        """
        ex1_ops, ex1_param_id, _ = self.get_ex1_ops()
        ex2_ops, ex2_param_id, _ = self.get_ex2_ops()

        ex_op = []
        param_ids = [-1]
        for _ in range(self.k):
            ex_op.extend(ex2_ops + ex1_ops)
            param_ids.extend([i + param_ids[-1] + 1 for i in ex2_param_id])
            param_ids.extend([i + param_ids[-1] + 1 for i in ex1_param_id])
        init_guess = np.random.rand(max(param_ids) + 1) - 0.5
        return ex_op, param_ids[1:], init_guess

    def get_ex1_ops(self, t1: np.ndarray = None) -> Tuple[List[Tuple], List[int], np.ndarray]:
        """
        Get generalized one-body excitation operators.

        Parameters
        ----------
        t1: np.ndarray, optional
            Not used. Kept for consistency with the parent method.

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: np.ndarray
            The initial guess for the parameters.

        See Also
        --------
        get_ex2_ops: Get generalized paired two-body excitation operators.
        get_ex_ops: Get one-body and two-body excitation operators for :math:`k`-UpCCGSD ansatz.
        """
        assert t1 is None
        no, nv = self.no, self.nv

        ex1_ops = []
        ex1_param_id = [-1]

        for a in range(no + nv):
            for i in range(a):
                # alpha to alpha
                ex_op_a = (no + nv + a, no + nv + i)
                # beta to beta
                ex_op_b = (a, i)
                ex1_ops.extend([ex_op_a, ex_op_b])
                ex1_param_id.extend([ex1_param_id[-1] + 1] * 2)

        ex1_init_guess = np.zeros(max(ex1_param_id) + 1)
        return ex1_ops, ex1_param_id[1:], ex1_init_guess

    def get_ex2_ops(self, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], np.ndarray]:
        """
        Get generalized paired two-body excitation operators.

        Parameters
        ----------
        t2: np.ndarray, optional
            Not used. Kept for consistency with the parent method.

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: np.ndarray
            The initial guess for the parameters.

        See Also
        --------
        get_ex1_ops: Get one-body excitation operators.
        get_ex_ops: Get one-body and two-body excitation operators for :math:`k`-UpCCGSD ansatz.
        """

        assert t2 is None
        no, nv = self.no, self.nv

        ex2_ops = []
        ex2_param_id = [-1]

        for a in range(no + nv):
            for i in range(a):
                # i correspond to a and j correspond to b, as in PySCF convention
                # otherwise the t2 amplitude has incorrect phase
                # paired
                ex_op_ab = (a, no + nv + a, no + nv + i, i)
                ex2_ops.append(ex_op_ab)
                ex2_param_id.append(ex2_param_id[-1] + 1)

        ex2_init_guess = np.zeros(max(ex2_param_id) + 1)
        return ex2_ops, ex2_param_id[1:], ex2_init_guess

    @property
    def e_kupccgsd(self):
        """
        Returns :math:`k`-UpCCGSD energy
        """
        return self.energy()
