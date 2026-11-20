import warnings

import pandas as pd
import numpy as np
from scipy import stats

from .harmonics import harmonic_basis

class PeriodicVAR:

    """Periodic VAR(p): A VAR whose coefficients change smoothly through the year.

    Every coefficient (intercept and lags) is a Fourier series in the day of year with `n_harm`
    harmonics, so the dynamics can differ between seasons. Optional exogenous variables enter on
    the same step with constant coefficients (build lagged or seasonal versions yourself as extra
    columns). Noise is Gaussian (via Cholesky) or bootstrapped from the residuals, with one
    covariance / residual pool per calendar month or one for the whole year.

    Notes
    -----
        - Estimation is OLS. With noise_pool="pooled" that is the exact Gaussian maximum likelihood.
        With "monthly" it is not (the exact MLE would weight each month by its own Σ), so loglike()
        is slightly below its true maximum.
        - loglike() and bic() always use the Gaussian likelihood, also with noise_draw="bootstrap".
        - Day of year runs 1..366: after 28 February the same day number is one calendar day later in
        leap years, and day 366 only occurs in leap years.
        - Typical use: PeriodicVAR(p=3).fit(data).simulate(30, seed=1)
    """

    # ---------- 1. Settings (chosen by you, never changed by fit) ----------
    def __init__(self, p=5, n_harm=2, freq="D", noise_draw="gaussian", noise_pool = "monthly"):
        """Store hyperparameters only (plus a check that the noise options exist).
            -p           number of lags
            -n_harm      harmonics per coefficient (0 = ordinary VAR with constant coefficients)
            -freq        frequency of the data
            -noise_draw  "gaussian" (normal noise via Cholesky of Σ) or "bootstrap" (resample residual rows)
            -noise_pool  "monthly" (one Σ / residual pool per calendar month) or "pooled" (one for the year)
        """
        self.p = p
        self.n_harm = n_harm
        self.freq = freq
        step = freq if freq[0].isdigit() else "1" + freq      # "h" -> "1h", "D" -> "1D", "3h" se queda igual
        self.dt = pd.Timedelta(step)
        self.noise_draw = noise_draw
        self.noise_pool = noise_pool

        NOISE_POOL_OPTIONS = {
            "monthly": "one covariance / residual pool per calendar month (noise size and correlation change through the year)",
            "pooled":  "one covariance / residual pool for the whole year (same noise every month)",
        }
        if noise_pool not in NOISE_POOL_OPTIONS:
            options = "\n".join(f"  - '{name}': {desc}" for name, desc in NOISE_POOL_OPTIONS.items())
            raise ValueError(f"noise_pool must be one of:\n{options}\nGot {noise_pool!r}.")

        NOISE_DRAW_OPTIONS = {
            "gaussian": "fit covariance matrix and apply cholesky decomposition to generate random noise",
            "bootstrap":  "bootstrap or resample noise",
        }
        if noise_draw not in NOISE_DRAW_OPTIONS:
            options = "\n".join(f"  - '{name}': {desc}" for name, desc in NOISE_DRAW_OPTIONS.items())
            raise ValueError(f"noise_draw must be one of:\n{options}\nGot {noise_draw!r}.")
    
    # ---------- 2. Fitting ----------
    def fit(self, data, exog=None, max_lag=None):
        """Estimate the model by OLS.

        data:    DataFrame on a DatetimeIndex, numeric.
        exog:    optional DataFrame of exogenous variables covering every fitting date (extra dates are
                 fine). They enter on the same step with constant coefficients. A model fitted with exog
                 needs exog again in forecast() and simulate().
        max_lag: drop the first `max_lag` (instead of the first p) before fitting. To compare models
        with different p by BIC, fit them all with the same max_lag so they use exactly the same time periods.
        Sets the learned attributes (trailing underscore) and returns self."""

        data = self._check_data(data)
        n_drop = self.p if max_lag is None else max_lag
        if n_drop < self.p:
            raise ValueError(f"max_lag={max_lag} must be at least p={self.p}")

        lag_matrix, Y = self._lag_matrix(data, self.p, max_lag=n_drop)   # same rows for every p with the same max_lag
        if len(Y) == 0:
            raise ValueError(f"no rows with all max_lag={n_drop} lags available: too many gaps/NaNs or too little data")
        self.max_lag_ = n_drop

        design_matrix = self._design(lag_matrix, self.n_harm)

        if exog is not None:
            if not isinstance(exog, pd.DataFrame):
                raise TypeError(f"exog must be a pandas DataFrame, got {type(exog).__name__}")
            if not isinstance(exog.index, pd.DatetimeIndex):
                raise TypeError("exog must have a DatetimeIndex")
            if exog.index.has_duplicates:
                raise ValueError("exog has duplicate dates")
            missing = design_matrix.index.difference(exog.index)
            if len(missing):
                raise ValueError(f"exog is missing {len(missing)} date(s) needed for fitting "
                                 f"(first {missing[0]:%Y-%m-%d %H:%M}, last {missing[-1]:%Y-%m-%d %H:%M})")
            clash = set(exog.columns) & (set(data.columns) | set(design_matrix.columns))
            if clash:
                raise ValueError(f"exog column names clash with existing columns: {sorted(clash)}")
            exog_part = exog.loc[design_matrix.index]          # exactly the fitting dates, same order
            nan_cols = exog_part.columns[exog_part.isna().any()].tolist()
            if nan_cols:
                raise ValueError(f"exog has NaNs on fitting dates in columns {nan_cols}")
            design_matrix = pd.concat([design_matrix, exog_part.astype(float)], axis=1)
            
        coefs, _, rank, _ = np.linalg.lstsq(design_matrix.values, Y.values, rcond=None)
        if rank < design_matrix.shape[1]:
            warnings.warn(f"Design matrix is rank deficient ({rank} of {design_matrix.shape[1]} columns): "
                          "coefficients are not unique. Check for constant or duplicated columns, or reduce p / n_harm.")

        self.coef_table_ = pd.DataFrame(coefs, index = design_matrix.columns, columns=Y.columns)

        resid = Y - design_matrix.values @ coefs
        self.resid_ = pd.DataFrame(data=resid, index=Y.index, columns=Y.columns)

        if exog is not None:
            self.exog_names_ = exog.columns.to_list()
            self.exog_coefs_ = self.coef_table_.loc[self.exog_names_].values
        else:
            self.exog_names_ = []
            self.exog_coefs_ = np.zeros((0, Y.shape[1]))
            
        self.columns_ = list(Y.columns)
        self.k_ = len(self.columns_)
        self.nobs_ = len(Y)
        self.last_obs_ = self._last_window(data)

        self.regressors_ = list(lag_matrix.columns)
        self.y_ = Y

        #### ---- For Gaussian Noise ----

        #### For noise_pool = pooled
        self.resid_pool_ = self.resid_.values
        self.sigma_pool_ = self.resid_.values.T @ self.resid_.values / len(self.resid_)
        self.chol_pool_ = self._cholesky(self.sigma_pool_, "the pooled residuals")

        #### For noise_pool = monthly
        self.sigma_month_ = {}
        self.chol_month_ = {}
        self.resid_month_ = {}
        for m in (range(1, 13) if self.noise_pool == "monthly" else []):
            r = self.resid_[self.resid_.index.month == m].values
            if len(r) <= self.k_:
                raise ValueError(f"Month {m} has only {len(r)} residuals; need more than k={self.k_} to estimate "
                                 "its covariance. Use more data or noise_pool='pooled'.")
            self.resid_month_[m] = r
            self.sigma_month_[m] = r.T @ r / len(r)
            self.chol_month_[m] = self._cholesky(self.sigma_month_[m], f"month {m}")

        #### Prebuild Coefs at DOY
        days = np.arange(1, 367)
        basis = harmonic_basis(days, n_harmonics=self.n_harm)   
        keys = list(basis)
        B = np.column_stack([np.broadcast_to(basis[key], days.shape) for key in keys])  # (366, n_basis)

        rows = [f"{r}_{key}" for r in self.regressors_ for key in keys]
        cube = self.coef_table_.loc[rows].values.reshape(len(self.regressors_), len(keys), self.k_)  # (n_reg, n_basis, k)

        self.daily_coefs_ = np.einsum("db,rbk->drk", B, cube)    # (366, n_reg, k)

        return self


    # Private helpers used by fit (leading underscore = internal)
    def _lag_matrix(self, df, lag, max_lag=None):
        """Lags built by TIME (multiples of self.dt), not by row, so gaps in the index are handled:
        a lag whose time step does not exist is NaN. Rows missing Y or any of the first `max_lag`
        lags are dropped (max_lag >= lag), so models with different p fitted with the same max_lag
        use exactly the same rows. Only the first `lag` lags are returned as regressors."""
        max_lag = lag if max_lag is None else max_lag
        LAG_MATRIX = df.copy()
        cols_origin = df.columns
        for i in range(max_lag):
            delay = df.shift(freq=(i + 1) * self.dt).reindex(df.index)
            for col in cols_origin:
                LAG_MATRIX[f'{col}_lag_{i+1}'] = delay[col].values
        LAG_MATRIX = LAG_MATRIX.dropna()
        Y = LAG_MATRIX[cols_origin]
        keep = [f'{col}_lag_{i+1}' for i in range(lag) for col in cols_origin]
        LAG_MATRIX = LAG_MATRIX[keep]
        LAG_MATRIX.insert(0, "const", 1.0)

        return LAG_MATRIX, Y

    def _last_window(self, data):
        """Last p consecutive time steps without NaNs (the start values for simulate/forecast)."""
        p = self.p
        full = data.reindex(pd.date_range(data.index[0], data.index[-1], freq=self.dt))
        complete = full.notna().all(axis=1).astype(int)
        ends = complete.rolling(p).sum() == p
        if not ends.any():
            raise ValueError(f"data has no stretch of p={p} consecutive steps without NaNs")
        end = ends[ends].index[-1]
        if end != data.index[-1]:
            warnings.warn(f"The data does not end with p={p} complete consecutive steps; start values "
                          f"for simulate/forecast are taken from the stretch ending {end:%Y-%m-%d %H:%M}.",
                          stacklevel=3)
        return full.loc[:end].iloc[-p:]

    def _check_data(self, data):
        """Raise a clear error unless `data` is a numeric DataFrame on a sorted DatetimeIndex whose
        dates fall on the `freq` grid. Gaps (missing steps) and NaNs are allowed: rows without all
        their lags are dropped later in _lag_matrix. Returns it as floats."""
        fmt = "%Y-%m-%d %H:%M"
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
        idx = data.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("data must have a DatetimeIndex")
        if idx.has_duplicates:
            raise ValueError(f"data has duplicate dates, e.g. {idx[idx.duplicated()][0]:{fmt}}")
        if not idx.is_monotonic_increasing:
            raise ValueError("data's dates must be sorted in increasing order (use data.sort_index())")
        # gaps are fine, but every jump must be a whole number of steps (dates on the freq grid)
        off_grid = np.flatnonzero((np.diff(idx.values) % self.dt.to_timedelta64()).astype("int64") != 0)
        if off_grid.size:
            i = off_grid[0]
            raise ValueError(f"data's dates are not on the {self.freq} grid: jump from {idx[i]:{fmt}} to "
                             f"{idx[i + 1]:{fmt}} is not a multiple of {self.dt} ({off_grid.size} such jump(s)). "
                             "Resample to a regular grid first.")
        non_numeric = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]
        if non_numeric:
            raise TypeError(f"data has non-numeric columns: {non_numeric}")
        constant = data.columns[data.nunique() <= 1].tolist()
        if constant:
            raise ValueError(f"data has constant columns {constant}: a variable that never changes can't be modelled")
        if "const" in data.columns:
            raise ValueError("'const' is reserved for the intercept: rename that column")
        if data.notna().all(axis=1).sum() <= self.p:
            raise ValueError(f"data needs more than p={self.p} complete rows (no NaNs)")
        return data.astype(float)

    @staticmethod
    def _cholesky(S, label):
        """Cholesky factor of S, with a readable error if S is not positive definite."""
        try:
            return np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            raise ValueError(f"The residual covariance for {label} is not positive definite: a variable "
                             "is probably constant or perfectly collinear with others there.") from None

    def _design(self, lag_matrix, n_harmonics): 

        basis = harmonic_basis(lag_matrix.index.dayofyear, n_harmonics=n_harmonics)

        cols = {}
        for col in lag_matrix.columns:
            for key, h in basis.items():
                cols[f"{col}_{key}"] = lag_matrix[col].values * h

        return pd.DataFrame(cols, index=lag_matrix.index)
    
    def coefs_at(self, doy):
        """Coefficient table for day of year `doy` (1..366): rows = intercept and lags, columns = equations.
        Exog coefficients are constant over the year and stored separately in exog_coefs_."""

        return pd.DataFrame(self.daily_coefs_[doy - 1], index=self.regressors_, columns=self.columns_)

    def _companion(self, d):
        """Companion matrix for day of year d (1..366)."""
        k, p = self.k_, self.p
        C = np.zeros((k * p, k * p))
        C[:k, :] = self.daily_coefs_[d - 1][1:1 + k * p].T   # skip const row -> [A_1 | ... | A_p]
        C[k:, :-k] = np.eye(k * (p - 1))
        return C

    def daily_roots(self):
        """Max |eigenvalue| of each day's companion matrix (Series, index 1..366).
        Diagnostic only: in a periodic VAR this is NOT a stability test."""
        return pd.Series({d: np.abs(np.linalg.eigvals(self._companion(d))).max()
                        for d in range(1, 367)}, name="max_root")

    def stability(self, period=365):
        """Periodic stability via the one-year monodromy matrix M = product of the companion matrices
        of every time step in the year. Coefficients change by day of year, so each day's companion
        is applied once per time step of that day (24 times with hourly data).
        stable iff rho < 1; rho_daily = rho**(1/period) is the average decay factor per day and
        rho_step = rho**(1/(period*steps_per_day)) the average decay factor per time step."""
        k, p = self.k_, self.p
        spd = max(1, int(round(pd.Timedelta("1D") / self.dt)))   # time steps per day (1 for daily data)
        M = np.eye(k * p)
        log_scale = 0.0
        for d in range(1, period + 1):
            M = np.linalg.matrix_power(self._companion(d), spd) @ M
            s = np.linalg.norm(M)          # renormalize each day so M never under/overflows
            M /= s
            log_scale += np.log(s)
        log_rho = log_scale + np.log(np.abs(np.linalg.eigvals(M)).max())
        return {"rho": float(np.exp(log_rho)),
                "rho_daily": float(np.exp(log_rho / period)),
                "rho_step": float(np.exp(log_rho / (period * spd))),
                "stable": bool(log_rho < 0)}

    def loglike(self):
        """Gaussian log-likelihood at the fitted Σ (monthly or pooled), also used for bootstrap models."""

        k = self.k_
        months = self.resid_.index.month
        ll = 0.0

        if self.noise_pool == 'monthly':
            for m in range(1, 13):
                n_m = (months == m).sum()
                logdet = np.linalg.slogdet(self.sigma_month_[m])[1]       # log of the determinant
                ll += -0.5 * n_m * (k * np.log(2 * np.pi) + logdet + k)
        elif self.noise_pool == 'pooled':
            logdet = np.linalg.slogdet(self.sigma_pool_)[1]
            ll += -0.5 * self.nobs_ * (k * np.log(2 * np.pi) + logdet + k)

        return ll

    def bic(self):
        """BIC = -2 loglike + n_params * log(nobs). Lower is better. To compare different p, fit all
        candidates with the same max_lag (see select_order) so they use the same steps."""
        return -2 * self.loglike() + self.n_params() * np.log(self.nobs_)

    def r2(self, adjusted=True):
        """R² per equation (one-step-ahead, in-sample)."""
        ss_res = (self.resid_ ** 2).sum()
        ss_tot = ((self.y_ - self.y_.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        if adjusted:
            n, q = self.nobs_, len(self.coef_table_)    # q = regressors per equation
            r2 = 1 - (1 - r2) * (n - 1) / (n - q)
        return r2
    
    def summary(self):
        """Readable overview of the fitted model. Returns a string (use print(model.summary()))."""
        st = self.stability()
        r2 = self.r2(adjusted=True)
        n_cov_blocks = 12 if self.noise_pool == "monthly" else 1
        w = 62
        line = "-" * w
        out = [
            "=" * w,
            "Periodic VAR".center(w),
            "=" * w,
            f"{'Variables (k)':<26}{self.k_:>8}    {', '.join(self.columns_)}",
            f"{'Frequency':<26}{self.freq:>8}    {self.dt}",
            f"{'Exogenous (q)':<26}{len(self.exog_names_):>8}    {', '.join(self.exog_names_) or '-'}",
            f"{'Lags (p)':<26}{self.p:>8}",
            f"{'Harmonics (n_harm)':<26}{self.n_harm:>8}",
            f"{'Noise':<26}{self.noise_draw + ' / ' + self.noise_pool:>20}",
            f"{'Sample':<26}{self.resid_.index[0]:%Y-%m-%d %H:%M} → {self.resid_.index[-1]:%Y-%m-%d %H:%M}",
            f"{'Observations (nobs)':<26}{self.nobs_:>8}",
            line,
            f"{'Coefficients':<26}{self.coef_table_.size:>8}    ({len(self.coef_table_)} per equation)",
            f"{'Covariance parameters':<26}{self.n_params() - self.coef_table_.size:>8}    ({n_cov_blocks} block(s))",
            f"{'Total parameters':<26}{self.n_params():>8}",
            f"{'Log-likelihood':<26}{self.loglike():>14.2f}",
            f"{'BIC':<26}{self.bic():>14.2f}",
            line,
            f"{'Stability (periodic)':<26}{'stable' if st['stable'] else 'UNSTABLE':>8}",
            f"{'  rho (one year)':<26}{st['rho']:>14.3e}",
            f"{'  rho_step':<26}{st['rho_step']:>14.4f}",
            line,
            f"{'Equation':<14}{'adj. R²':>10}{'resid std':>12}{'lag-1 autocorr':>16}",
        ]
        for c in self.columns_:
            out.append(f"{c:<14}{r2[c]:>10.3f}{self.resid_[c].std():>12.3f}{self.resid_[c].autocorr(1):>16.3f}")
        out.append("=" * w)
        return "\n".join(out)

    def n_params(self):
        """number of estimated parameters: coefficients + covariance entries"""
        if self.noise_pool == 'monthly':
            n_cov = 12 * self.k_ * (self.k_ + 1) // 2
        elif self.noise_pool == 'pooled':
            n_cov = self.k_ * (self.k_ + 1) // 2
        return self.coef_table_.size + n_cov

    #### Residual checks
    def diagnostics(self, lags=None):
        """Residual checks, one row per variable.

        acf_h            autocorrelation of the residuals at lag h (should be close to 0)
        LB_resid_*       Ljung-Box test up to max(lags) on the residuals: a small p-value means leftover
                         step-to-step correlation, so the model misses something (try a larger p or n_harm)
        LB_squared_*     the same test on squared residuals: a small p-value means volatility clustering
                         (calm and wild spells), so shocks are not independent from step to step
        Degrees of freedom = max(lags), as statsmodels' acorr_ljungbox with model_df=0."""
        if lags is None:
            spd = int(pd.Timedelta("1D") / self.dt)        # steps per day
            lags = (1, 7, 14, 30) if spd <= 1 else (1, spd, 2 * spd, 7 * spd)
        R = self.resid_
        n, L = len(R), max(lags)
        rows = {}
        for c in self.columns_:
            x = R[c].values
            row = {f"acf_{h}": self._acf(x, h) for h in lags}
            for name, series in (("resid", x), ("squared", x ** 2)):
                Q = n * (n + 2) * sum(self._acf(series, h) ** 2 / (n - h) for h in range(1, L + 1))
                row[f"LB_{name}_Q"] = Q
                row[f"LB_{name}_pvalue"] = stats.chi2.sf(Q, L)
            rows[c] = row
        return pd.DataFrame(rows).T

    @staticmethod
    def _acf(x, h):
        """Sample autocorrelation of x at lag h (same definition as statsmodels' acf)."""
        d = x - x.mean()
        return float(d[h:] @ d[:-h] / (d @ d))
    
    # ---------- 4. Using the fitted model ----------

    def _prepare_exog(self, exog, dates):
        """Check exog against the fitted model and return it as an array (len(dates), q),
        one row per date in `dates`, columns in the order of self.exog_names_."""
        if not self.exog_names_:                                   # model fitted without exog
            if exog is not None:
                raise ValueError("This model was fitted without exog, so exog must not be passed.")
            return np.zeros((len(dates), 0))

        if exog is None:
            raise ValueError(f"This model was fitted with exog {self.exog_names_}: "
                             "pass exog covering the requested dates.")
        if not isinstance(exog, pd.DataFrame):
            raise TypeError(f"exog must be a pandas DataFrame, got {type(exog).__name__}")
        if not isinstance(exog.index, pd.DatetimeIndex):
            raise TypeError("exog must have a DatetimeIndex")
        if exog.index.has_duplicates:
            raise ValueError("exog has duplicate dates")
        missing_cols = [c for c in self.exog_names_ if c not in exog.columns]
        if missing_cols:
            raise ValueError(f"exog is missing the model's exog columns {missing_cols}")
        missing_dates = dates.difference(exog.index)
        if len(missing_dates):
            raise ValueError(f"exog is missing {len(missing_dates)} of the requested date(s) "
                             f"(first {missing_dates[0]:%Y-%m-%d %H:%M}, last {missing_dates[-1]:%Y-%m-%d %H:%M})")
        X = exog.loc[dates, self.exog_names_]                      # right rows, right column order
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            raise ValueError(f"exog has NaNs on the requested dates in columns {nan_cols}")
        return X.to_numpy(dtype=float)

    def _run(self, start_values, dates, mode, exog=None, rng=None, noise=None):
        """Core step-by-step recursion shared by forecast and simulate.

        start_values: (p, k) array, the p steps before steps[0], oldest first
        dates:        DatetimeIndex of the steps to produce (decides day-of-year coefficients and month)
        mode:         "forecast" (no noise) or "simulate" (adds noise, needs rng)
        exog:         (len(steps), q) array, one row per date; q = 0 without exog
        Returns an array (len(steps), k), start values not included."""

        if mode not in ("forecast", "simulate"):
            raise ValueError(f"mode must be 'forecast' or 'simulate', got {mode!r}")
        if mode == "simulate" and rng is None:
            raise ValueError("mode='simulate' needs a random generator (rng)")
        
        k, p = self.k_, self.p

        z = np.zeros((p + len(dates), k))
        z[:p] = start_values

        for s, date in enumerate(dates):
            t = p + s                                   # row in z for this step

            # the past p simulated steps (+ the 1 for the intercept)
            lag_row = np.concatenate([[1.0]] + [z[t - l] for l in range(1, p + 1)])

            # predictable part: look up today's coefficients by day of year
            deterministic = lag_row @ self.daily_coefs_[date.dayofyear-1]

            if mode == 'simulate':
                # random part: get, generate or resample residuals
                if noise is not None:
                    eps = noise[s]
                elif self.noise_draw == 'bootstrap':
                    if self.noise_pool == 'monthly':
                        eps = rng.choice(self.resid_month_[date.month], axis=0)
                    elif self.noise_pool =='pooled':
                        eps = rng.choice(self.resid_pool_, axis=0)
                elif self.noise_draw == 'gaussian':
                    if self.noise_pool == 'monthly':
                        eps = self.chol_month_[date.month] @ rng.standard_normal(k)
                    elif self.noise_pool =='pooled':
                        eps = self.chol_pool_ @ rng.standard_normal(k)

            # today's value, stored so tomorrow can use it as "yesterday"
            z[t] = deterministic + exog[s] @ self.exog_coefs_

            if mode == 'simulate':
                z[t] = z[t] + eps

        return z[p:]

    def forecast(self, steps, exog=None, history=None):
        """Deterministic forecast for `steps` steps: the expected path, no noise.

        Starts the step after the last step of `history` (default: the end of the training data).
        Only the last p rows of history are used. If the model has exog, pass exog covering the
        forecast steps."""
        p = self.p

        if history is None:
            history = self.last_obs_
        else:
            if not isinstance(history, pd.DataFrame):
                raise TypeError(f"history must be a pandas DataFrame, got {type(history).__name__}")
            if not isinstance(history.index, pd.DatetimeIndex):
                raise TypeError("history must have a DatetimeIndex (dates decide the day-of-year coefficients)")
            missing = [c for c in self.columns_ if c not in history.columns]
            if missing:
                raise ValueError(f"history is missing the model's columns {missing}")
            if len(history) < p:
                raise ValueError(f"history needs at least p={p} rows, got {len(history)}")
            if len(history) > p:
                warnings.warn(f"history has {len(history)} rows; only the last p={p} are used.", stacklevel=2)
                history = history.iloc[-p:]
            history = history[self.columns_]          # same column order as the model
            if history.isna().any().any():
                raise ValueError("the last p rows of history contain NaNs")
            
        first = pd.Timestamp(history.index[-1]) + self.dt
        dates = pd.date_range(first, periods=steps, freq=self.dt)

        X = self._prepare_exog(exog, dates)
        forecast = self._run(start_values=history.values, dates=dates, mode='forecast', exog=X, rng=None, noise=None)

        return pd.DataFrame(forecast, index=dates, columns=self.columns_)

    def simulate(self, n_years=None, start=None, seed=None, burn=None, exog=None, noise=None):
        """Stochastic simulation. Returns a DataFrame with dates.

        The simulated dates come from, in this order:
          - exog.index, if the model was fitted with exog (continuous steps; no n_years/start);
          - noise.index, if precomputed noise is passed (continuous steps; no n_years/start);
          - otherwise n_years starting at `start` (default "2001-01-01").
        noise: optional DataFrame of precomputed innovations, one row per simulated date and one
               column per variable (k). It replaces the internal draw, so noise_draw, noise_pool and
               seed are ignored. With exog too, noise.index must equal exog.index.
        The burn-in steps before the first date reuse the first `burn` rows of exog and/or noise."""
        if burn is None:
            burn = int(pd.Timedelta(days=365) / self.dt)

        if noise is not None:
            if not isinstance(noise, pd.DataFrame):
                raise TypeError(f"noise must be a pandas DataFrame, got {type(noise).__name__}")
            if not isinstance(noise.index, pd.DatetimeIndex):
                raise TypeError("noise must have a DatetimeIndex")
            if noise.shape[1] != self.k_:
                raise ValueError(f"noise must have k={self.k_} columns ({self.columns_}), got {noise.shape[1]}")
            if noise.isna().any().any():
                raise ValueError("noise contains NaNs")

        if self.exog_names_:
            if exog is None:
                raise ValueError(f"This model was fitted with exog {self.exog_names_}: pass exog; "
                                 "its index decides the simulated dates.")
            if n_years is not None or start is not None:
                raise ValueError("With exog, the simulated dates are taken from exog.index: "
                                 "do not pass n_years or start.")
            out_dates = exog.index
            source = "exog"
        else:
            if exog is not None:
                raise ValueError("This model was fitted without exog, so exog must not be passed.")
            if noise is not None:
                if n_years is not None or start is not None:
                    raise ValueError("With noise, the simulated dates are taken from noise.index: "
                                     "do not pass n_years or start.")
                out_dates = noise.index
                source = "noise"
            else:
                if n_years is None:
                    raise ValueError("n_years is required when the model has no exog and no noise is given")
                start = "2001-01-01" if start is None else start
                n_steps = int(round(pd.Timedelta(days=365.25 * n_years) / self.dt))
                out_dates = pd.date_range(start, periods=n_steps, freq=self.dt)
                source = None

        if source is not None:
            if not isinstance(out_dates, pd.DatetimeIndex):
                raise TypeError(f"{source} must have a DatetimeIndex")
            if len(out_dates) > 1 and (np.diff(out_dates.values) != self.dt.to_timedelta64()).any():
                raise ValueError(f"{source}.index must be continuous steps (sorted, no gaps) to simulate on it")
            if burn > len(out_dates):
                raise ValueError(f"burn={burn} is longer than {source} ({len(out_dates)} steps), whose first "
                                 "rows are reused for the burn-in: pass more steps or use a smaller burn")

        if noise is not None:
            if not noise.index.equals(out_dates):
                raise ValueError("noise.index must equal the simulated dates (exog.index when the model has exog)")
            N_out = noise[self.columns_].to_numpy(dtype=float) if set(self.columns_) <= set(noise.columns) \
                else noise.to_numpy(dtype=float)
            N = np.vstack([N_out[:burn], N_out])             # burn-in reuses its first rows, as exog
        else:
            N = None

        # exog for the output days, then reuse its first `burn` rows for the burn-in
        X_out = self._prepare_exog(exog, out_dates)                    # (n_out, q); q = 0 without exog
        X_burn = X_out[:burn] if self.exog_names_ else np.zeros((burn, 0))
        X = np.vstack([X_burn, X_out])

        # dates: `burn` spin-up steps right before the first output step, then the output steps
        burn_dates = pd.date_range(end=out_dates[0] - self.dt, periods=burn, freq=self.dt)
        dates = burn_dates.append(out_dates)

        rng = np.random.default_rng(seed)
        sim = self._run(start_values=self.last_obs_.values, dates=dates, mode='simulate', exog=X, rng=rng, noise=N)

        return pd.DataFrame(sim[burn:], index=out_dates, columns=self.columns_)

def select_order(data, p_values, exog=None, **model_kwargs):
    """Fit PeriodicVAR for every p in p_values on the same steps (max_lag = max(p_values))
    and return a table of nobs, log-likelihood, number of parameters and BIC. Lowest BIC = best."""
    p_values = sorted(p_values)
    max_lag = max(p_values)
    rows = []
    for p in p_values:
        m = PeriodicVAR(p=p, **model_kwargs).fit(data, exog=exog, max_lag=max_lag)
        rows.append({"p": p, "nobs": m.nobs_, "loglike": m.loglike(), "n_params": m.n_params(), "bic": m.bic()})
    out = pd.DataFrame(rows).set_index("p")
    out["best"] = out["bic"] == out["bic"].min()
    return out