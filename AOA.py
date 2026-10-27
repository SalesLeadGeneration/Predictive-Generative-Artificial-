import time
import numpy as np


def AOA(X, obj_func, lb, ub, iterations):
    """
    obj_func        → objective function accepting vector x
    dim             → dimension of decision variables
    lb, ub          → lower / upper bounds (scalar or list)
    pop_size        → number of addaxes
    iterations      → number of iterations
    constraint_func → function returning list of constraint values g(x) ≤ 0 satisfied
    minimize        → True for minimization
    seed            → random seed
    """
    constraint_func = None
    minimize = True
    seed = None
    verbose = False
    pop_size, dim = X.shape[0], X.shape[1]
    if seed is not None:
        np.random.seed(seed)

    lb = np.asarray(lb if np.ndim(lb) else np.repeat(lb, dim)).astype(float)
    ub = np.asarray(ub if np.ndim(ub) else np.repeat(ub, dim)).astype(float)

    # Objective values
    F = np.full(pop_size, np.inf if minimize else -np.inf)
    ct = time.time()

    def evaluate(x):
        f = obj_func(x)

        if constraint_func is not None:
            cons = constraint_func(x)
            penalty = sum(max(0, c) for c in cons)

            if penalty > 0:
                f = f + 1e8 * penalty  # heavy penalty

        return f

    for i in range(pop_size):
        F[i] = evaluate(X[i])

    # Best
    if minimize:
        best_F = np.min(F)
        best_X = X[np.argmin(F)].copy()
    else:
        best_F = np.max(F)
        best_X = X[np.argmax(F)].copy()

    history = []

    for t in range(1, iterations + 1):

        for i in range(pop_size):

            xi = X[i].copy()
            fi = F[i]

            better = [k for k in range(pop_size) if F[k] < fi]

            if len(better) > 0:
                SA = X[np.random.choice(better)]
            else:
                SA = best_X.copy()

            r1 = np.random.rand(dim)
            Iij = np.random.choice([1, 2], size=dim)

            x_p1 = xi + r1 * (SA - Iij * xi)
            x_p1 = np.clip(x_p1, lb, ub)

            f_p1 = evaluate(x_p1)

            # Eq. (6) accept if better
            if minimize:
                if f_p1 <= fi:
                    xi, fi = x_p1, f_p1
            else:
                if f_p1 >= fi:
                    xi, fi = x_p1, f_p1

            r2 = np.random.rand(dim)
            dig = (1 - 2 * r2) * ((ub - lb) / t)

            x_p2 = xi + dig
            x_p2 = np.clip(x_p2, lb, ub)

            f_p2 = evaluate(x_p2)

            # Eq. (8) accept if better
            if minimize:
                if f_p2 <= fi:
                    xi, fi = x_p2, f_p2
            else:
                if f_p2 >= fi:
                    xi, fi = x_p2, f_p2

            # Update population
            X[i] = xi
            F[i] = fi

            # Global best update
            if minimize:
                if fi < best_F:
                    best_F = fi
                    best_X = xi.copy()
            else:
                if fi > best_F:
                    best_F = fi
                    best_X = xi.copy()

        history.append(best_F)

        if verbose and (t % (iterations // 10) == 0):
            print(f"Iteration {t}/{iterations}  Best = {best_F}")
    ct = time.time() - ct
    return best_F, history, best_X, ct
