"""Weather-type-conditioned periodic VAR.

    z(t) = c(t) + d[k(t)] + A_1(t) z(t-1) + ... + A_p(t) z(t-p) + e(t),   e(t) ~ N(0, Sigma_month)

- c(t), A_l(t): harmonics of the day of year (same as PeriodicVAR)
- d[k]:         one shift per weather type k (replaces the constant's base term)
- k(t):         type of the target day t (from the raw PCs); ERA5 when fitting, ALR when simulating

With wt=None (or one single type) it is the plain periodic VAR.
"""
import numpy as np
import pandas as pd

from tv_var.harmonics import harmonic_basis


class WTPeriodicVAR:

    def __init__(self, p=5, n_harm=2):
        self.p = p
        self.n_harm = n_harm

    # ------------------------------------------------------------------ helpers
    def _lags(self, z):
        """[1, z(t-1), ..., z(t-p)] for every t >= p, built on the continuous series."""
        z = np.asarray(z, dtype=float)
        n = len(z)
        X = [np.ones(n - self.p)] + [z[self.p - l:n - l] for l in range(1, self.p + 1)]
        return np.column_stack(X)                                   # (n-p, 1 + k*p)

    def _harm(self, doy):
        return np.column_stack(list(harmonic_basis(doy, self.n_harm).values()))   # (n, 1 + 2H)

    # ------------------------------------------------------------------ fit
    def fit(self, z_df, wt=None):
        """z_df: standardized PCs (DataFrame, daily). wt: Series of types (ints) on the same index, or None."""
        self.names_ = list(z_df.columns)
        k, p = z_df.shape[1], self.p
        Y   = z_df.values[p:]                                       # targets
        dates = z_df.index[p:]
        L   = self._lags(z_df.values)                               # (n, 1 + k*p), column 0 = const
        Hm  = self._harm(dates.dayofyear)                           # (n, nb), column 0 = base
        nb  = Hm.shape[1]

        # every regressor x every harmonic: (n, (1+k*p)*nb), ordered regressor-major
        X = (L[:, :, None] * Hm[:, None, :]).reshape(len(Y), -1)

        if wt is None:
            wt = pd.Series(0, index=z_df.index)
        wt_t = wt.reindex(dates).values                             # type of each TARGET day
        assert not pd.isna(wt_t).any(), "wt has missing days"
        self.types_ = np.unique(wt_t).astype(int)
        D = (wt_t[:, None] == self.types_[None, :]).astype(float)   # (n, K) type indicators

        # drop const x base (column 0): replaced by the K type indicators
        X = np.hstack([D, X[:, 1:]])
        B = np.linalg.lstsq(X, Y, rcond=None)[0]

        K = len(self.types_)
        self.d_ = pd.DataFrame(B[:K], index=self.types_, columns=self.names_)          # (K, k)
        rest = np.vstack([np.zeros((1, k)), B[K:]])                                     # const x base = 0
        self.coef_ = rest.reshape(1 + k * p, nb, k)                                     # (regressor, harmonic, pc)

        self.resid_ = pd.DataFrame(Y - X @ B, index=dates, columns=self.names_)
        self.nobs_, self.n_params_ = len(Y), B.size
        months = dates.month
        self.sigma_ = {m: np.cov(self.resid_.values[months == m].T, bias=True) for m in range(1, 13)}
        self.chol_  = {m: np.linalg.cholesky(s) for m, s in self.sigma_.items()}
        self.first_obs_ = z_df.values[:p]
        self._tables = {}
        return self

    # ------------------------------------------------------------------ daily coefficients
    def table_at(self, doy):
        """(1 + k*p, k): coefficients for one day of year (const row excludes the type shift)."""
        if doy not in self._tables:
            h = self._harm([doy])[0]                                # (nb,)
            self._tables[doy] = np.einsum("rhk,h->rk", self.coef_, h)
        return self._tables[doy]

    def stability(self):
        k, p = len(self.names_), self.p
        out = {}
        for doy in range(1, 367):
            A = self.table_at(doy)[1:].T                            # (k, k*p)
            C = np.zeros((k * p, k * p)); C[:k] = A; C[k:, :-k] = np.eye(k * (p - 1))
            out[doy] = np.abs(np.linalg.eigvals(C)).max()
        return pd.Series(out)

    # ------------------------------------------------------------------ simulate
    def simulate(self, wt_seq, seed=None, init=None):
        """wt_seq: Series of types indexed by the simulation dates. Starts from `init` (p x k), default the
        first p fitted days, with no burn-in. Returns standardized z on wt_seq's dates."""
        rng = np.random.default_rng(seed)
        k, p = len(self.names_), self.p
        dates = wt_seq.index
        d = self.d_.reindex(wt_seq.values).values                  # (n, k) shift of each day
        assert not np.isnan(d).any(), "wt_seq has types not seen in the fit"

        z = np.zeros((p + len(dates), k))
        z[:p] = self.first_obs_ if init is None else init
        doys, months = dates.dayofyear, dates.month
        for i in range(len(dates)):
            t = p + i
            x = np.concatenate([[1.0]] + [z[t - l] for l in range(1, p + 1)])
            z[t] = x @ self.table_at(doys[i]) + d[i] + self.chol_[months[i]] @ rng.standard_normal(k)
        return pd.DataFrame(z[p:], index=dates, columns=self.names_)
