#  Copyright (c) 2024. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

import numpy as np

from scipy.optimize import OptimizeResult


def soap(fun, x0, args=(), maxfev=2000, callback=None, **kwargs):
    """
    Scipy Optimizer interface for sequantial optimization with
    approximate parabola (SOAP)

    Parameters
    ----------
    fun : callable ``f(x, *args)``
        Function to be optimized.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function.
    maxfev : int
        Maximum number of function evaluations to perform.
        Default: 2000.
    callback : callable, optional
        Called after each iteration.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a SciPy ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array.
    """

    nfev = 0
    nit = 0

    def _fun(_x):
        nonlocal nfev
        nfev += 1
        return fun(_x, *args)

    trajectory = [x0.copy()]
    vec_list = []
    # direction order
    metric = np.abs(x0)
    for i in np.argsort(metric)[::-1]:
        vec = np.zeros_like(x0)
        vec[i] = 1
        vec_list.append(vec)
    vec_list_copy = vec_list.copy()

    e_list = [_fun(trajectory[-1])]
    offset_list = []
    diff_list = []

    scale = 0.1

    while nfev < maxfev:
        if len(vec_list) != 0:
            vec = vec_list[0]
            vec_list = vec_list[1:]
        else:
            vec_list = vec_list_copy.copy()
            # continue
            if len(trajectory) < len(vec_list_copy):
                continue
            p0 = trajectory[-1 - len(vec_list_copy)]
            f0 = e_list[-1 - len(vec_list_copy)]
            pn = trajectory[-1]
            fn = e_list[-1]
            fe = _fun(2 * pn - p0)
            if fe > f0:
                continue
            average_direction = pn - p0
            if np.allclose(average_direction, 0):
                continue
            average_direction /= np.linalg.norm(average_direction)
            replace_idx = np.argmax(np.abs(diff_list[-len(vec_list_copy) :]))
            df = np.abs(diff_list[-len(vec_list_copy) :][replace_idx])
            if 2 * (f0 - 2 * fn + fe) * (f0 - fn - df) ** 2 > (f0 - fe) ** 2 * df:
                continue
            del vec_list[replace_idx]
            vec_list = [average_direction] + vec_list
            vec_list_copy = vec_list.copy()
            continue

        vec_normed = vec / np.linalg.norm(vec)
        x = [-scale, 0, scale]
        es = [None, e_list[-1], None]
        for j in [0, -1]:
            es[j] = _fun(trajectory[-1] + x[j] * vec_normed)
        if np.argmin(es) == 0:
            x = [-scale * 4, -scale, 0, scale]
            es = [None, es[0], es[1], es[2]]
            for j in [0]:
                es[j] = _fun(trajectory[-1] + x[j] * vec_normed)
        elif np.argmin(es) == 2:
            x = [-scale, 0, scale, scale * 4]
            es = [es[0], es[1], es[2], None]
            for j in [-1]:
                es[j] = _fun(trajectory[-1] + x[j] * vec_normed)
        else:
            assert np.argmin(es) == 1
        a, b, c = np.polyfit(x, es, 2)
        if np.argmin(es) not in [0, 3]:
            offset = b / 2 / a
            e = c - b**2 / 4 / a
        else:
            # print(a, b)
            offset = -x[np.argmin(es)]
            e = np.argmin(es)
        offset_list.append(offset)
        trajectory.append(trajectory[-1] - offset * vec_normed)
        if len(es) == 3:
            e_list.append(e)
        else:
            e_list.append(_fun(trajectory[-1]))
        diff_list.append(e_list[-1] - e_list[-2])

        if callback is not None:
            callback(np.copy(x0))

        nit += 1

    return OptimizeResult(fun=e_list[-1], x=trajectory[-1], nit=nit, nfev=nfev, success=True)
