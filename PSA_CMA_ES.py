import numpy as np

class PSA_CMA_ES:
    """
    CMA-ES with Population Size Adaptation (PSA-CMA-ES)
    based on "PSA-CMA-ES: CMA-ES with Population Size Adaptation"
    (Nishida & Akimoto, GECCO 2018).
    """
    def __init__(self, dim, objective, init_mean=None, init_sigma=0.3,
                 lambda_min=None, lambda_max=None,
                 alpha=1.4, beta=0.4,
                 max_evals=1e5, tol=1e-12):
        self.dim = dim
        self.f = objective
        # Initialization
        self.m = np.zeros(dim) if init_mean is None else np.array(init_mean)
        self.sigma = init_sigma
        # Strategy parameters
        self.lambda_def = 4 + int(3 * np.log(dim))
        self.lambda_min = self.lambda_def if lambda_min is None else lambda_min
        self.lambda_max = np.inf if lambda_max is None else lambda_max
        self.alpha = alpha
        self.beta = beta
        # State
        self.C = np.eye(dim)
        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        self.p_theta = np.zeros(dim * (dim + 3) // 2)
        self.gamma_sigma = 0.0
        self.gamma_c = 0.0
        self.gamma_theta = 0.0
        self.lambda_real = float(self.lambda_def)
        self.lambda_int = self.lambda_def
        # Other
        self.max_evals = int(max_evals)
        self.tol = tol
        self.eval_count = 0

    def _update_weights(self, lam):
        mu = lam // 2
        weights = np.array([np.log(mu + 0.5) - np.log(i + 1)
                             for i in range(mu)])
        weights /= np.sum(weights)
        w = np.zeros(lam)
        w[:mu] = weights
        mu_eff = 1.0 / np.sum(weights**2)
        return w, mu, mu_eff

    def _recombination(self, X, w):
        return X.T.dot(w)

    def _vech(self, M):
        # upper-triangular including diagonal
        idx = np.triu_indices_from(M)
        return M[idx]

    def _ivech(self, v):
        # reconstruct symmetric matrix from vech
        n = self.dim
        M = np.zeros((n, n))
        idx = np.triu_indices(n)
        M[idx] = v
        M[(idx[1], idx[0])] = v
        return M

    def _fisher_sqrt(self):
        # Fisher information for N(m, Sigma): block diag( Sigma^{-1}, 1/2 * (Sigma^{-1} ⊗ Sigma^{-1}) )
        # We need sqrt: multiply movement vector by sqrt of block
        # For m: C^{-1/2}/sigma
        # For Sigma: we flatten and apply appropriate scaling, but approximate by identity since normalization divides by expectation.
        # Here we construct a block diagonal sqrt Fisher factor for simplicity.
        # Compute C^{-1/2}
        eigvals, eigvecs = np.linalg.eigh(self.C)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        C_inv_sqrt = eigvecs.dot(D_inv_sqrt).dot(eigvecs.T)
        # Block for m part
        A_m = C_inv_sqrt / self.sigma
        # Block for Sigma part (vech): approximate by identity scaled by 1/(sqrt(2)) / sigma^2
        # This is a simplification; full implementation requires Kronecker.
        scale = 1.0 / (self.sigma**2 * np.sqrt(2))
        A_S = scale * np.eye(len(self.p_theta))
        return A_m, A_S

    def optimize(self):
        while self.eval_count < self.max_evals:
            # (Re)compute parameters based on current lambda_int
            w, mu, mu_eff = self._update_weights(self.lambda_int)
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((self.dim + 2)**2 + mu_eff))
            # Sample
            B = np.linalg.cholesky(self.C)
            Z = np.random.randn(self.lambda_int, self.dim)
            X = self.m + self.sigma * Z.dot(B.T)
            # Evaluate
            F = np.apply_along_axis(self.f, 1, X)
            self.eval_count += self.lambda_int
            # Sort
            idx = np.argsort(F)
            X_sel = X[idx[:mu]]
            Z_sel = Z[idx[:mu]]
            # Update m
            m_old = self.m.copy()
            self.m = self._recombination(X_sel, w[:mu])
            delta_m = self.m - m_old
            # Update evolution paths p_sigma and p_c
            C_inv_sqrt = np.linalg.inv(B).T
            y = C_inv_sqrt.dot(delta_m) / self.sigma
            self.p_sigma = (1 - c_sigma) * self.p_sigma + \
                np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            norm_p_sigma = np.linalg.norm(self.p_sigma)
            # Heaviside for p_c
            chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 2/(self.dim+1)) * chi_n else 0.0
            self.p_c = (1 - c_c) * self.p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_m / self.sigma
            # Update C
            rank_one = np.outer(self.p_c, self.p_c) - self.C
            rank_mu = sum(w[i] * (np.outer(Z_sel[i], Z_sel[i]) - self.C) for i in range(mu))
            self.C = self.C + c1 * rank_one + c_mu * rank_mu
            # Update sigma
            self.sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1))
            # ---- Population size adaptation ----
            # Compute delta_theta
            # Flatten Sigma change
            Sigma_old = (self.sigma**2) * self.C
            # After sigma and C update
            Sigma_new = (self.sigma**2) * self.C
            delta_Sigma = Sigma_new - Sigma_old
            delta_theta = np.concatenate([delta_m, self._vech(delta_Sigma)])
            # Normalize via Fisher sqrt
            A_m, A_S = self._fisher_sqrt()
            u_m = A_m.dot(delta_m)
            u_S = A_S.dot(self._vech(delta_Sigma))
            u = np.concatenate([u_m, u_S])
            # Update p_theta
            self.p_theta = (1 - self.beta) * self.p_theta + \
                np.sqrt(self.beta * (2 - self.beta)) * u / np.linalg.norm(u)
            # Update gamma_theta
            self.gamma_theta = (1 - self.beta)**2 * self.gamma_theta + self.beta * (2 - self.beta)
            # Adjust lambda_real
            self.lambda_real *= np.exp(self.beta * (self.gamma_theta - np.dot(self.p_theta, self.p_theta) / self.alpha))
            self.lambda_real = np.clip(self.lambda_real, self.lambda_min, self.lambda_max)
            self.lambda_int = int(round(self.lambda_real))
            # Step-size correction placeholder (requires precomputed optimal scaling)
            # For simplicity, omitted; can be computed per eq. (17)-(18).
            # Termination
            if np.linalg.norm(delta_m) < self.tol:
                break
        return self.m

# Example usage:
# def sphere(x): return np.sum(x**2)
# optimizer = PSA_CMA_ES(dim=10, objective=sphere)\# result = optimizer.optimize()
# print(result)
