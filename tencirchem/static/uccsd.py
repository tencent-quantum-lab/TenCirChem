#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from typing import Tuple, List

import numpy as np
from pyscf.gto.mole import Mole

from tencirchem.static.ucc import UCC
from tencirchem.constants import DISCARD_EPS


class UCCSD(UCC):
    """
    Run UCCSD calculation. For a comprehensive tutorial see :doc:`/tutorial_jupyter/ucc_functions`.

    Examples
    --------
    >>> import numpy as np
    >>> from tencirchem import UCCSD
    >>> from tencirchem.molecule import h2
    >>> uccsd = UCCSD(h2)
    >>> e_ucc = uccsd.kernel()
    >>> np.testing.assert_allclose(e_ucc, uccsd.e_fci, atol=1e-10)
    >>> e_hf = uccsd.energy(np.zeros(uccsd.n_params))
    >>> np.testing.assert_allclose(e_hf, uccsd.e_hf, atol=1e-10)
    """

    def __init__(
        self,
        mol: Mole,
        init_method: str = "mp2",
        active_space: Tuple[int, int] = None,
        mo_coeff: np.ndarray = None,
        pick_ex2: bool = True,
        epsilon: float = DISCARD_EPS,
        sort_ex2: bool = True,
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
        pick_ex2: bool, optional
            Whether screen out two body excitations based on the inital guess amplitude.
            Defaults to True, which means excitations with amplitude less than ``epsilon`` (see below) are discarded.
        epsilon: float, optional
            The threshold to discard two body excitations. Defaults to 1e-12.
        sort_ex2: bool, optional
            Whether sort two-body excitations in the ansatz based on the initial guess amplitude.
            Large excitations come first. Defaults to True.
            Note this could lead to different ansatz for the same molecule at different geometry.
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
        tencirchem.KUPCCGSD
        tencirchem.PUCCD
        tencirchem.UCC
        """
        super().__init__(
            mol,
            init_method,
            active_space,
            mo_coeff,
            engine=engine,
            run_hf=run_hf,
            run_mp2=run_mp2,
            run_ccsd=run_ccsd,
            run_fci=run_fci,
        )
        self.pick_ex2 = pick_ex2
        self.sort_ex2 = sort_ex2
        # screen out excitation operators based on t2 amplitude
        self.t2_discard_eps = epsilon
        self.ex_ops, self.param_ids, self.init_guess = self.get_ex_ops(self.t1, self.t2)

    def get_ex_ops(self, t1: np.ndarray = None, t2: np.ndarray = None) -> Tuple[List[Tuple], List[int], List[float]]:
        """
        Get one-body and two-body excitation operators for UCCSD ansatz.
        Pick and sort two-body operators if ``self.pick_ex2`` and ``self.sort_ex2`` are set to ``True``.

        Parameters
        ----------
        t1: np.ndarray, optional
            Initial one-body amplitudes based on e.g. CCSD
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
        get_ex2_ops: Get two-body excitation operators.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> ex_op, param_ids, init_guess = uccsd.get_ex_ops()
        >>> ex_op
        [(3, 2), (1, 0), (1, 3, 2, 0)]
        >>> param_ids
        [0, 0, 1]
        >>> init_guess  # doctest:+ELLIPSIS
        [0.0, ...]
        """
        ex1_ops, ex1_param_ids, ex1_init_guess = self.get_ex1_ops(t1)
        ex2_ops, ex2_param_ids, ex2_init_guess = self.get_ex2_ops(t2)

        # screen out symmetrically not allowed excitation
        ex2_ops, ex2_param_ids, ex2_init_guess = self.pick_and_sort(
            ex2_ops, ex2_param_ids, ex2_init_guess, self.pick_ex2, self.sort_ex2
        )

        ex_op = ex1_ops + ex2_ops
        param_ids = ex1_param_ids + [i + max(ex1_param_ids) + 1 for i in ex2_param_ids]
        init_guess = ex1_init_guess + ex2_init_guess
        return ex_op, param_ids, init_guess

    def pick_and_sort(self, ex_ops, param_ids, init_guess, do_pick=True, do_sort=True):
        # sort operators according to amplitude
        if do_sort:
            sorted_ex_ops = sorted(zip(ex_ops, param_ids), key=lambda x: -np.abs(init_guess[x[1]]))
        else:
            sorted_ex_ops = list(zip(ex_ops, param_ids))
        ret_ex_ops = []
        ret_param_ids = []
        for ex_op, param_id in sorted_ex_ops:
            # discard operators with tiny amplitude.
            # The default eps is so small that the screened out excitations are probably not allowed
            if do_pick and np.abs(init_guess[param_id]) < self.t2_discard_eps:
                continue
            ret_ex_ops.append(ex_op)
            ret_param_ids.append(param_id)
        unique_ids = np.unique(ret_param_ids)
        ret_init_guess = np.array(init_guess)[unique_ids]
        id_mapping = {old: new for new, old in enumerate(unique_ids)}
        ret_param_ids = [id_mapping[i] for i in ret_param_ids]
        return ret_ex_ops, ret_param_ids, list(ret_init_guess)

    @property
    def e_uccsd(self) -> float:
        """
        Returns UCCSD energy
        """
        return self.energy()
