import time
import numpy as np


def DOA(X, objective, lb, ub, max_iter):
    """
    Full Dollmaker Optimization Algorithm (DOA)
    Implements all Equations (1) to (7) exactly as in the PDF.

    objective : function  -> fitness function
    lb, ub    : scalars or arrays -> lower/upper bounds
    dim       : dimension (required if lb, ub are scalars)
    """
    minimize = True
    seed = None
    pop_size, dim = X.shape[0], X.shape[1]
    rng = np.random.default_rng(seed)

    if np.isscalar(lb):
        if dim is None:
            raise ValueError("dim required for scalar bounds")
        lb = np.full(dim, float(lb))
        ub = np.full(dim, float(ub))
    else:
        lb = np.array(lb, float)
        ub = np.array(ub, float)
        dim = len(lb)

    F = np.array([objective(ind) for ind in X], dtype=float)

    # Minimization or Maximization handler
    def better(a, b):
        return a < b if minimize else a > b

    # Find initial best doll
    best_idx = np.argmin(F) if minimize else np.argmax(F)
    best_pos = X[best_idx].copy()
    best_val = F[best_idx]

    history = [best_val]

    ct = time.time()
    for t in range(1, max_iter + 1):

        for i in range(pop_size):

            xi = X[i].copy()

            P = best_pos.copy()  # Doll pattern (best solution)
            I = rng.choice([1, 2])  # Random integer {1, 2}
            r = rng.random()  # Random scalar [0,1]

            x_p1 = xi + r * (P - I * xi)  # Eq. (4)

            # Apply bound constraints
            x_p1 = np.minimum(np.maximum(x_p1, lb), ub)

            f_p1 = objective(x_p1)

            if better(f_p1, F[i]):
                X[i] = x_p1
                F[i] = f_p1

                if better(f_p1, best_val):
                    best_val = f_p1
                    best_pos = x_p1.copy()

            r_ij = rng.random(dim)  # Per-dimension random numbers

            step = (1 - 2 * r_ij) * (ub - lb) / float(t)
            x_p2 = X[i].copy() + step  # Eq. (6)

            # Enforce boundaries
            x_p2 = np.minimum(np.maximum(x_p2, lb), ub)

            f_p2 = objective(x_p2)

            if better(f_p2, F[i]):
                X[i] = x_p2
                F[i] = f_p2

                if better(f_p2, best_val):
                    best_val = f_p2
                    best_pos = x_p2.copy()

        history.append(best_val)

    ct = time.time() - ct
    return best_val, history, best_pos, ct
