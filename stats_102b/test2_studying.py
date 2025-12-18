import numpy as np

def gd_ridge(beta=None, X=None, y=None,
             lam=1.0,
             eta=0.01,
             tol=1e-8,
             max_iter=1000):

    n, p = X.shape
    beta = np.zeros(p) if beta is None else beta.copy()

    for _ in range(max_iter):
        y_pred = X @ beta
        error = y_pred - y

        # Gradient of ridge loss
        grad = (1 / n) * (X.T @ error) + lam * beta

        beta_new = beta - eta * grad

        if np.linalg.norm(beta_new - beta) < tol:
            break

        beta = beta_new

    return beta

def soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(0, np.abs(z) - gamma)

def gd_lasso(beta=None, X=None, y=None,
             lam=1.0,
             eta=1e-3,
             tol=1e-8,
             max_iter=1000):

    n, p = X.shape
    beta = np.zeros(p) if beta is None else beta.copy()

    for _ in range(max_iter):
        grad = (1 / n) * (X.T @ (X @ beta - y))

        z = beta - eta * grad
        beta_new = soft_threshold(z, eta * lam)

        if np.linalg.norm(beta_new - beta) < tol:
            break

        beta = beta_new

    return beta


def newton_sse(beta=None, X=None, y=None,
               tol=1e-8,
               max_iter=100,
               verbose=False):

    n, p = X.shape
    beta = np.zeros(p) if beta is None else beta.copy()

    for i in range(max_iter):
        beta_old = beta.copy()

        grad = (1 / n) * (X.T @ (X @ beta - y))
        hess = (1 / n) * (X.T @ X)

        step = np.linalg.solve(hess, grad)
        beta = beta - step

        if np.linalg.norm(beta - beta_old) < tol:
            if verbose:
                print(f"Converged in {i+1} iterations")
            break

    return beta


def coordinate_descent_lm(beta=None, X=None, y=None,
                          tol=1e-6,
                          max_iter=1000):

    n, p = X.shape
    beta = np.zeros(p) if beta is None else beta.copy()

    for _ in range(max_iter):
        beta_new = beta.copy()

        for j in range(p):
            r_j = y - X[:, np.arange(p) != j] @ beta_new[np.arange(p) != j]
            beta_new[j] = np.sum(r_j * X[:, j]) / np.sum(X[:, j] ** 2)

        if np.linalg.norm(beta_new - beta) < tol:
            break

        beta = beta_new

    return beta


def f(beta, X, y, lam=0.0):
    n = X.shape[0]
    return (1 / (2*n)) * np.sum((X @ beta - y)**2) + lam * np.sum(np.abs(beta))


def grad_f(beta, X, y):
    n = X.shape[0]
    return (X.T @ (X @ beta - y)) / n


def gd_backtracking(beta=None, X=None, y=None,
                    eta=1.0, ep=0.5, tau=0.5,
                    tol=1e-6, max_iter=1000):

    beta = np.zeros(X.shape[1]) if beta is None else beta.copy()

    for _ in range(max_iter):
        grad = grad_f(beta, X, y)
        eta_bt = eta

        while f(beta - eta_bt * grad, X, y) > \
              f(beta, X, y) - ep * eta_bt * np.sum(grad**2):
            eta_bt *= tau

        beta_new = beta - eta_bt * grad

        if np.linalg.norm(beta_new - beta) < tol:
            break

        beta = beta_new

    return beta

def gd_lasso_backtracking(beta=None, X=None, y=None,
                          lam=1.0,
                          eta=1.0, ep=0.5, tau=0.5,
                          tol=1e-6, max_iter=1000):

    n, p = X.shape
    beta = np.zeros(p) if beta is None else beta.copy()

    for _ in range(max_iter):
        grad = grad_f(beta, X, y)
        eta_bt = eta

        z = beta - eta_bt * grad
        gamma = eta_bt * lam

        while f(soft_threshold(z, gamma), X, y, lam) > \
              f(beta, X, y, lam) - ep * eta_bt * np.sum(grad**2):
            eta_bt *= tau
            z = beta - eta_bt * grad
            gamma = eta_bt * lam

        beta_new = soft_threshold(z, gamma)

        if np.linalg.norm(beta_new - beta) < tol:
            break

        beta = beta_new

    return beta