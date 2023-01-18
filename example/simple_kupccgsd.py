import numpy as np

from tencirchem import KUPCCGSD
from tencirchem.molecule import h4

# setup
# number of different initial guesses
n_tries = 10
kupccgsd = KUPCCGSD(h4, n_tries=n_tries)
# calculate
kupccgsd.kernel()
# analyze result
kupccgsd.print_summary()


# use custom initial guess
kupccgsd.init_guess_list = np.random.rand(n_tries, kupccgsd.n_params)
kupccgsd.kernel()
kupccgsd.print_summary()
