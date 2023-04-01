===============
Advanced Topics
===============


Engines
-------
TenCirChem offers a set of different engines for the simulation of static molecular properties
and is summarized below.

.. list-table::
    :header-rows: 1

    * - Engine
      - Compatible class
      - State representation
      - UCC factor expansion
      - Note
    * - ``"civector"``
      - ``UCC``
      - CI vector
      - Yes
      - Most efficient
    * - ``"civector-large"``
      - ``UCC``
      - CI vector
      - Yes
      - Efficient and memory-friendly
    * - ``"tensornetwork"``
      - ``UCC``, ``HEA``
      - statevector
      - No
      - Noiseless tensor network contraction
    * - ``"tensornetwork-noise"``
      - ``HEA``
      - density matrix
      - No
      - with gate noise
    * - ``"tensornetwork-shot"``
      - ``HEA``
      - statevector
      - No
      - with measurement noise
    * - ``"tensornetwork-noise&shot"``
      - ``HEA``
      - density matrix
      - No
      - with gate and measurement noise
    * - ``"statevector"``
      - ``UCC``
      - statevector
      - Yes
      - Experimental
    * - ``"pyscf"``
      - ``UCC``
      - CI vector
      - Yes
      - Experimental

"CI vector" means Configuration Interaction vector.
It is a more efficient representation than statevector for chemical applications
because CI vector exploits particle number conservation symmetry.
This is one of the major reasons why TenCirChem is so fast.

"UCC factor expansion" means expanding the excitation operator exponential into a polynomial form

.. math::

     e^{\theta G} = I + G^2 - \cos\theta G^2 +\sin \theta G

where :math:`G` is the excitation operator

.. math::

    G = \begin{cases}
        a^\dagger_i a_j - a^\dagger_j a_i , \\
        a^\dagger_i a^\dagger_j a_k a_l - a^\dagger_l a^\dagger_k a_j a_i
        \end{cases}

The expansion allows fast simulation of :math:`e^{\theta G}`,
without decomposing :math:`e^{\theta G}` into quantum gates.
It is not difficult to derive the equation noticing that

.. math::

    G^3 = - G

In the special case of :math:`G^2=-I`, the famous formula :math:`e^{\theta G}=\cos\theta + \sin\theta G` is recovered.
The idea was proposed by the
`Fermionic Quantum Emulator <https://quantum-journal.org/papers/q-2021-10-27-568/>`__
based on lots of preceding works.
UCC factor expansion is another major reason why TenCirChem is so fast.


For UCC tasks, ``"civector"`` engine is most efficient,
however, it is also memory consuming due to cached intermediates.
``"civector-large"`` is similar to ``"civector"`` except that caching is only used in a limited scope.
In general, it is recommended to use the ``"civector"`` engine until the host/GPU memory runs out,
and then switch to the ``"civector-large"`` engine.

In the :class:`UCC <tencirchem.UCC>` class ``"civector"`` is the default engine for calculations with <= 16 qubits
and ``"civector-large"`` is used for larger scale calculation.
In the :class:`HEA <tencirchem.HEA>` class the noiseless ``"tensornetwork"`` engine is used by default.

To override the default engine, pass the string to the ``engine`` argument of the ``UCC`` class:

.. code-block:: python

    from tencirchem import UCCSD
    from tencirchem.molecule import h4
    uccsd = UCCSD(h4, engine="civector-large")
    print(uccsd.kernel())
    print(uccsd.energy(engine="tensornetwork"))


Backends
--------
TenCirChem supports NumPy, CuPy and JAX backend, in combination with a set of different `Engines`_.
Note that the supported backends differ slightly between TenCirChem and TensorCircuit.

* To simplify installation, the default backend is NumPy.
* When calculation scales up and GPU is available, CuPy becomes an efficient replacement of NumPy.
* JAX offers auto-differentiation (AD) and just-in-time compilation (JIT) over NumPy and CuPy.
  It is the only available backend for the dynamics module.
  For static algorithms, JAX is sometimes faster than NumPy and CuPy due to JIT compilation.

For UCC tasks, TenCirChem typically does not rely on the AD for gradients
and JIT time is usually much longer than the actual run time.
Sometimes JIT will also result in out-of-memory error.
Thus, in general NumPy and CuPy are the recommended backends over JAX.

For noisy circuit simulation, the performance of NumPy and JAX is comparable.
For quantum dynamics simulation, the JAX backend is usually preferred.

TenCirChem shares the same API with TensorCircuit for setting backends and data types, at runtime:

.. code-block:: python

    from tencirchem import set_backend, set_dtype
    set_backend("numpy")
    set_dtype("complex64")

TenCirChem by default uses ``complex128`` data type, which is different from the TensorCircuit convention.
``complex128`` is highly recommended with NumPy backend or ``float64``-friendly GPU.


Boson Encoding
--------------

An important step in quantum dynamics simulation is to map bosons into qubits.
TenCirChem supports 4 different ways to encode boson DOFs into qubits.

* No encoding. A boson DOF is directly truncated to a two-level system.
* Unary encoding. Use one qubit to represent one boson level.
* Binary encoding. Uses bitstring to represent one boson level.
* Gray code encoding. An improved version of binary encoding.

A detailed description of the unary encoding and gray encoding, with other variants, can be found in
`this article <https://www.nature.com/articles/s41534-020-0278-0>`__ .

In the following table we make a quick comparison of the encodings for a 4-level boson.

.. list-table::
    :header-rows: 1

    * - Boson level
      - No encoding
      - Unary encoding
      - Binary encoding
      - Gray encoding
    * - 0
      - 0
      - 0001
      - 00
      - 00
    * - 1
      - 1
      - 0010
      - 01
      - 01
    * - 2
      - Truncated
      - 0100
      - 10
      - 11
    * - 3
      - Truncated
      - 1000
      - 11
      - 10

Unary encoding costs :math:`\mathcal{O}(N)` qubits where :math:`N` is boson level,
yet the circuit is shallower.
Gray encoding costs :math:`\mathcal{O}(\textrm{log}N)` qubits, yet the circuit depth is deeper.
Gray encoding is strictly better than binary encoding.

The ``TimeEvolution`` class by default uses Gray code encoding.
Other encodings can be specified through the ``boson_encoding`` argument of the ``TimeEvolution`` class.
