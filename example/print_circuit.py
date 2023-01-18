from tencirchem import UCCSD
from tencirchem.molecule import h2

# setup
uccsd = UCCSD(h2)
# TensorCircuit object
circuit = uccsd.get_circuit()
# optimized circuit described in https://arxiv.org/pdf/2005.14475.pdf
print("circuit:")
print(circuit.draw())
# The original flavour Trotter decomposition circuit
print("original circuit:")
print(uccsd.get_circuit(trotter=True).draw(fold=256))
