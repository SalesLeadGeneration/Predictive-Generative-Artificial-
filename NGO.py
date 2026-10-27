import numpy as np
import time


def NGO(X, objective,  Lowerbound, Upperbound, Max_iterations):
    Search_Agents, dimensions = X.shape[0], X.shape[1]
    ct = time.time()

    Lowerbound = np.ones(dimensions) * Lowerbound
    Upperbound = np.ones(dimensions) * Upperbound

    X_new = np.zeros((Search_Agents, dimensions))
    fit = np.zeros(Search_Agents)
    fit_new = np.zeros(Search_Agents)
    NGO_curve = np.zeros(Max_iterations)

    for i in range(dimensions):
        X[:, i] = Lowerbound[i] + np.random.rand(Search_Agents) * (Upperbound[i] - Lowerbound[i])

    for i in range(Search_Agents):
        L = X[i, :]
        fit[i] = objective(L)

    for t in range(Max_iterations):
        best = np.min(fit)
        blocation = np.argmin(fit)

        if t == 0:
            xbest = X[blocation, :].copy()
            fbest = best
        elif best < fbest:
            fbest = best
            xbest = X[blocation, :].copy()

        for i in range(Search_Agents):
            # Phase 1: Exploration
            I = round(1 + np.random.rand())
            k = np.random.choice(Search_Agents, 1)[0]
            P = X[k, :]
            F_P = fit[k]

            if fit[i] > F_P:
                X_new[i, :] = X[i, :] + np.random.rand(dimensions) * (P - I * X[i, :])
            else:
                X_new[i, :] = X[i, :] + np.random.rand(dimensions) * (X[i, :] - P)

            X_new[i, :] = np.maximum(X_new[i, :], Lowerbound)
            X_new[i, :] = np.minimum(X_new[i, :], Upperbound)

            L = X_new[i, :]
            fit_new[i] = objective(L)
            if fit_new[i] < fit[i]:
                X[i, :] = X_new[i, :]
                fit[i] = fit_new[i]

            # Phase 2: Exploitation
            R = 0.02 * (1 - (t + 1) / Max_iterations)
            X_new[i, :] = X[i, :] + (-R + 2 * R * np.random.rand(dimensions)) * X[i, :]

            X_new[i, :] = np.maximum(X_new[i, :], Lowerbound)
            X_new[i, :] = np.minimum(X_new[i, :], Upperbound)

            L = X_new[i, :]
            fit_new[i] = objective(L)
            if fit_new[i] < fit[i]:
                X[i, :] = X_new[i, :]
                fit[i] = fit_new[i]

        best_so_far = fbest
        average = np.mean(fit)
        Score = fbest
        Best_pos = xbest
        NGO_curve[t] = Score
    ct = time.time() - ct

    return Score, NGO_curve,  Best_pos, ct