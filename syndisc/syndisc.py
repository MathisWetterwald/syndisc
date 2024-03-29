"""
Synergistic disclosure and self-disclosure in discrete random variables.

References:

    F. Rosas*, P. Mediano*, B. Rassouli and A. Barrett (2019). An operational
    information decomposition via synergistic disclosure.

    B. Rassouli, Borzoo, F. Rosas, and D. Gündüz (2018). Latent Feature
    Disclosure under Perfect Sample Privacy. In 2018 IEEE WIFS, pp. 1-7.

Distributed under the modified BSD licence. See LICENCE for details.

Pedro Mediano and Fernando Rosas, 2019
"""
import numpy as np
import dit

def build_constraint_matrix(cons, d):
    """
    Build constraint matrix.

    The constraint matrix is a matrix P that is the vertical stack
    of all the preserved marginals.

    Parameters
    ----------
    cons : iter of iter
        List of variable indices to preserve.
    d : dit.Distribution
        Distribution for which to design the constraints

    Returns
    -------
    P : np.ndarray
        Constraint matrix

    """
    # Initialise a uniform distribution to make sure it has full support
    u = dit.distconst.uniform_like(d)
    n = len(u.rvs)
    l = u.rvs
    u = u.coalesce(l + l)

    # Generate one set of rows of P per constraint
    P_list = []
    for c in cons:
        pX123, pX1gX123 = u.condition_on(crvs=range(n, 2*n), rvs=c)

        pX123.make_dense()
        for p in pX1gX123:
          p.make_dense()

        P_list.append(np.hstack([p.pmf[:,np.newaxis] for p in pX1gX123]))

    # Stack rows and return
    P = np.vstack(P_list)

    return P

