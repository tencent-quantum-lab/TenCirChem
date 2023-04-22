=============
Quick Start
=============
TenCirChem contains two primary modules: the static module and the dynamic module,
responsible for molecular static and dynamic properties respectively.
In the static module, the :class:`UCC <tencirchem.UCC>` classes is for noiseless UCC circuit simulation
and the :class:`HEA <tencirchem.HEA>` class is for noisy simulation or arbitrary circuits.


Installation
------------
TenCirChem can be installed via ``pip``

.. code-block:: sh

    pip install tencirchem

This will install a minimal version of TenCirChem with NumPy backend.
For GPU support and some of the advanced features,
install `JAX <https://github.com/google/jax>`__ and `CuPy <https://github.com/cupy/cupy>`__ for respective backends.

TenCirChem relies heavily on `PySCF <https://pyscf.org/index.html>`__, which for the moment does not support Windows platform.
Windows users are recommended to install TenCirChem in the `WSL <https://learn.microsoft.com/en-us/windows/wsl/>`__ environment.


UCC Calculation
---------------
For static molecular properties using UCC ansatze,
TenCirChem has implemented
:class:`UCCSD <tencirchem.UCCSD>` (`Peruzzo2014 <https://www.nature.com/articles/ncomms5213>`__),
:class:`kUpCCGSD <tencirchem.KUPCCGSD>` (`Lee2019 <https://pubs.acs.org/doi/10.1021/acs.jctc.8b01004>`__),
and
:class:`pUCCD <tencirchem.PUCCD>` (`Elfving2021 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.032605>`__),
based on the :class:`UCC <tencirchem.UCC>` base class.

.. literalinclude:: ../../example/ucc_classes.py
    :linenos:


Apart from the high-level interface inherited from PySCF,
TenCirChem offers a rich set of useful intermediate interfaces for access and modification,
to facilitate the development and validation of novel algorithms.
Functions of the :class:`UCC <tencirchem.UCC>` classes include

* outputting the Hamiltonian as integrals or OpenFermion objects,
* outputting the circuit as TensorCircuit object,
* active space approximation,
* calculating one and two body reduced density matrices,
* interfacing with PySCF for CASSCF calculation and nuclear gradients,

and much more. Please refer to the
:doc:`tutorial_jupyter/ucc_functions` tutorial for detailed guide and
the `Examples <https://github.com/tencent-quantum-lab/TenCirChem/tree/master/example>`__ directory
for working examples.


UCC Speed
---------
The :class:`UCC <tencirchem.UCC>` class exploits particle number conservation
and *UCC factor expansion* for extremely efficient simulation.
Here we show the tested UCCSD wall time of TenCirChem over hydrogen chain system,
along with corresponding error compared to FCI.
The bond length is 0.8 Å and the basis set is STO-3G.
We do not include the wall time of other packages, but the tremendous speed up is quite visible.

.. list-table::
    :header-rows: 1

    * - Molecule
      - Qubits
      - Circuit Depth
      - Parameters
      - Wall time (s)
      - Error (mH)
    * - :math:`\rm{H}_4`
      - 8
      - 78
      - 11
      - 0.05
      - 0.01
    * - :math:`\rm{H}_6`
      - 12
      - 462
      - 39
      - 0.4
      - 0.27
    * - :math:`\rm{H}_8`
      - 16
      - 1,834
      - 108
      - 1.9
      - 0.72
    * - :math:`\rm{H}_{10}`
      - 20
      - 5,586
      - 246
      - 14
      - 1.37
    * - :math:`\rm{H}_{12}`
      - 24
      - 13,917
      - 495
      - 51
      - 2.13
    * - :math:`\rm{H}_{14}`
      - 28
      - 30,428
      - 899
      - 2093
      - 2.92
    * - :math:`\rm{H}_{16}`
      - 32
      - 59,634
      - 1,520
      - 50909
      - 3.72

For the first three row the engine is ``"civector"`` with NumPy backend,
and the rest is with ``"civector-large"`` engine and CuPy backend on V100 GPU.
For more details please refer to :ref:`faq:Why is TenCirChem so fast?`.

Noisy Circuit Simulation
------------------------
TenCirChem supports noisy circuit simulation through the :class:`HEA <tencirchem.HEA>` class.
:func:`HEA.ry <tencirchem.HEA.ry>` accepts integrals and setups the class with :math:`R_y` ansatz.
The ``engine`` argument controls how the energy is evaluated.
``"tensornetwork"`` is for noiseless simulation,
``"tensornetwork-noise"`` includes gate noise,
``"tensornetwork-shot"`` includes measurement uncertainty,
and ``"tensornetwork-noise&shot"`` includes both.

.. literalinclude:: ../../example/noisy_circuit.py
    :linenos:

The :class:`HEA <tencirchem.HEA>` class supports arbitrary circuit and Hamiltonian.
The cutomization guide is included in the
:doc:`tutorial_jupyter/noisy_simulation` tutorial.


Dynamics Calculation
--------------------
In the following we show how to perform quantum dynamics simulation using a 1-mode spin-boson model as an example.

.. math::

    \hat H = \epsilon \hat \sigma_z + \Delta \hat \sigma_x + \omega \hat b^\dagger \hat b
              + g \sigma_z (\hat b^\dagger + \hat b)

For simplicity in the following we assume :math:`\epsilon=\Delta=\omega=g=1`.


TenCirChem uses the ``Op`` and ``BasisSet`` classes in `Renormalizer <https://github.com/shuaigroup/Renormalizer>`__
to define the Hamiltonian and the basis sets.


.. code-block:: python

    # equivalent to `from renormalizer import ...`
    from tencirchem import Op, BasisHalfSpin, BasisSHO

    ham_terms = [
        Op("sigma_z", "spin"),
        Op("sigma_x", "spin"),
        Op(r"b^\dagger b", "boson"),
        Op("sigma_z", "spin") * Op(r"b^\dagger+b", "boson")
    ]
    basis = [BasisHalfSpin("spin"), BasisSHO("boson", omega=1, nbas=4)]


For more about these classes please refer to the
`Renormalizer document <https://shuaigroup.github.io/Renormalizer/model.html>`__.

Next, run the time evolution

.. code-block:: python

    from tencirchem import TimeEvolution, set_backend

    # the dynamics module only supports JAX backend
    set_backend("jax")

    # property_op_dict determines the physical observables that is calculated during time evolution
    te = TimeEvolution(ham_terms, basis, property_op_dict={"Z": Op("sigma_z", spin)})
    # one step of time evolution and time step is 0.1
    te.kernel(0.1)
    # print physical observable
    print(te.properties["Z"])


See the :doc:`tutorial_jupyter/sbm_dynamics` for what's happening under the hood and how to customize the ``TimeEvolution`` class.

Further Readings
----------------
That's all for the basics of TenCirChem!
If you wish to learn more, you may visit

* The :doc:`advanced` for advanced features and the internal mechanisms.
* The :doc:`tutorials` as Jupyter Notebook.
* The :doc:`modules`.
* The `GitHub repository <https://github.com/tencent-quantum-lab/TenCirChem>`__ for source code and recent developments.