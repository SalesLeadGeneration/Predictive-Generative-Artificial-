import numpy as np, math, time, sys


def PROPOSED(X, obj_func, lb, ub, max_iter):
    """
    Spider-tailed Horned Viper Optimization (STHVO) with Modified Exploration (Eq.5).

    Parameters
    ----------
    X : np.ndarray
        Initial population, shape (pop_size, dim)
    obj_func : callable
        Objective function to minimize: obj_func(x: np.ndarray) -> float
    lb, ub : float or array-like
        Lower and upper bounds
    max_iter : int
        Maximum iterations

    Returns
    -------
    best_f : float
        Best objective value found
    history : dict
        Tracking dictionary with keys: best, mean, std, worst
    best_x : np.ndarray
        Best solution vector
    total_time : float
        Total computation time in seconds
    """
    pop_size, dim = X.shape[0], X.shape[1]
    seed = None
    verbose = True
    stagnation_tol = 50
    rng = np.random.default_rng(seed)

    # --- Convert bounds to arrays ---
    def _to_array(b):
        a = np.array(b, dtype=float)
        if a.size == 1:
            return np.repeat(a.item(), dim)
        return a

    lb = _to_array(lb)
    ub = _to_array(ub)
    if np.any(ub <= lb):
        raise ValueError("Every upper bound must be > lower bound")

    # --- Safe objective wrapper ---
    def _safe_obj(x):
        try:
            val = float(obj_func(np.array(x, dtype=float)))
            if np.isnan(val) or np.isinf(val):
                return np.finfo(float).max / 10.0
            return val
        except Exception as e:
            if verbose:
                print(f"[sthvo] Objective error: {e}", file=sys.stderr)
            return np.finfo(float).max / 10.0

    # --- Initialize fitness ---
    F = np.array([_safe_obj(X[i]) for i in range(pop_size)], dtype=float)
    best_idx = int(np.argmin(F))
    best_x = X[best_idx].copy()
    best_f = F[best_idx]

    history = {"best": [best_f],
               "mean": [float(np.mean(F))],
               "std": [float(np.std(F))],
               "worst": [float(np.max(F))]}

    start_time = time.time()
    stagn_counter = 0
    best_seen = best_f

    for t in range(1, max_iter + 1):

        # --- Phase 1: Modified Exploration (Proposed Eq.5) ---
        for i in range(pop_size):
            Fi = F[i]  # Current fitness
            candidates = np.where(F <= Fi)[0]
            k = int(rng.choice(candidates))
            SA = X[k].copy()  # Selected viper position
            I = int(rng.choice([1, 2]))

            # Fitness metrics
            mean_f = float(np.mean(F))
            worst_f = float(np.max(F))
            best_f_curr = best_f

            Xp1 = X[i] + math.sin(
                math.pi / 2 * (worst_f - Fi + best_f_curr) / (Fi + best_f_curr + mean_f ** 2)    # PROPOSED Updation
            ) * (SA - I * X[i])

            Xp1 = np.clip(Xp1, lb, ub)
            Fp1 = _safe_obj(Xp1)

            if Fp1 < Fi:
                X[i] = Xp1
                F[i] = Fp1

        # --- Phase 2: Exploitation (Eq.7-Eq.8) ---
        for i in range(pop_size):
            r = float(rng.random())
            delta = (1 - 2 * math.sin(math.pi / 2 * r)) * ((ub - lb) / float(t))
            Xp2 = np.clip(X[i] + delta, lb, ub)
            Fp2 = _safe_obj(Xp2)
            if Fp2 < F[i]:
                X[i] = Xp2
                F[i] = Fp2

        # --- Update best ---
        cur_best_idx = int(np.argmin(F))
        cur_best_f = F[cur_best_idx]
        if cur_best_f < best_f:
            best_f = cur_best_f
            best_x = X[cur_best_idx].copy()

        # --- History tracking ---
        history["best"].append(best_f)
        history["mean"].append(float(np.mean(F)))
        history["std"].append(float(np.std(F)))
        history["worst"].append(float(np.max(F)))

        # --- Stagnation check ---
        if best_f < best_seen - 1e-15:
            stagn_counter = 0
            best_seen = best_f
        else:
            stagn_counter += 1
        if stagnation_tol and stagn_counter >= stagnation_tol:
            if verbose:
                print(f"[STHVO] Stopping early due to stagnation at iter {t}")
            break

        if verbose and (t == 1 or t % max(1, max_iter // 20) == 0 or t == max_iter):
            print(
                f"[Iter {t}/{max_iter}] best={best_f:.6e} mean={history['mean'][-1]:.6e} std={history['std'][-1]:.6e}"
            )

    total_time = time.time() - start_time
    if verbose:
        print(f"[STHVO] Finished in {t} iters, time={total_time:.3f}s, best={best_f:.12e}")

    return best_f, history, best_x, total_time
