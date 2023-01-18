from tencirchem import UCCSD, set_backend
from tencirchem.molecule import h4

# use the default numpy backend, or switch to any other backends
set_backend("numpy")

# setup
uccsd = UCCSD(h4)
# calculate
uccsd.kernel()
# analyze result
uccsd.print_summary(include_circuit=True)
