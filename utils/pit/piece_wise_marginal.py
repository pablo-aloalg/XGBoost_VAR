import numpy as np
import pandas as pd
from scipy.stats import norm

def to_normal(p, eps=1e-10):
    """Probabilidades (PIT) -> Z normal estándar. Recorta p a [eps, 1-eps] para evitar ±inf."""
    z = norm.ppf(np.clip(np.asarray(p, dtype=float), eps, 1 - eps))
    if isinstance(p, pd.Series):
        return pd.Series(z, index=p.index, name=p.name)
    return z


def from_normal(z):
    """Z normal estándar -> probabilidades en (0, 1)."""
    p = norm.cdf(np.asarray(z, dtype=float))
    if isinstance(z, pd.Series):
        return pd.Series(p, index=z.index, name=z.name)
    return p

class PiecewiseMarginal:
    def __init__(self, pieces, quantiles):
        self.pieces = pieces
        self.quantiles = quantiles

        if len(pieces) != len(quantiles) + 1:
            raise ValueError("Number of pieces must be one more than the number of quantiles.")
        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("Quantiles must be between 0 and 1.")
        if not all(quantiles[i] < quantiles[i + 1] for i in range(len(quantiles) - 1)):
            raise ValueError("Quantiles must be in increasing order.")

    def fit(self, data):
        # Fit each piece to the corresponding segment of the data

        if isinstance(data, pd.Series):
            self.index = data.index
        else:
            self.index = None

        data = np.asarray(data, dtype=float)
        self.mask = ~np.isnan(data)
        self.x = data[self.mask]          # quita los NaN
        self.n = len(self.x)

        if self.n == 0:
            raise ValueError("PiecewiseMarginal.fit: no hay datos válidos (todo NaN o vacío).")

        
        # Determine the thresholds based on the quantiles
        self.thresholds = np.quantile(self.x, self.quantiles)
        self.labels = np.searchsorted(self.thresholds, self.x, side="left")
        self.bounds = np.concatenate(([-np.inf], self.thresholds, [np.inf]))

        for i, piece in enumerate(self.pieces):
            piece_data = self.x[self.labels == i]
            if len(piece_data) > 0:
                piece.fit(piece_data, lower=self.bounds[i], upper=self.bounds[i+1])
            else:
                raise ValueError(f"PiecewiseMarginal.fit: no hay datos válidos para el segmento {i}.")

        self.counts = np.bincount(self.labels, minlength=len(self.pieces))
        self.weights = self.counts / self.n
        self.cdf_thresholds = np.concatenate(([0], np.cumsum(self.weights)))

        return self

    def pit(self):
        # PIT en muestra: probabilidades globales de los datos del ajuste
        p = np.empty(self.n)
        for i, piece in enumerate(self.pieces):
            in_piece = self.labels == i
            p_local = piece.pit_in_sample()
            p[in_piece] = self.cdf_thresholds[i] + self.weights[i] * p_local

        # volver al tamaño original, con NaN donde los datos eran NaN
        p_full = np.full(self.mask.shape, np.nan)
        p_full[self.mask] = p

        if self.index is not None:
            return pd.Series(p_full, index=self.index)
        return p_full

    def cdf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        p = np.full_like(x, np.nan)

        for i, piece in enumerate(self.pieces):
            in_piece = (x > self.bounds[i]) & (x <= self.bounds[i + 1])
            p_local = piece.cdf(x[in_piece])
            p[in_piece] = self.cdf_thresholds[i] + self.weights[i] * p_local

        return p

    def ppf(self, p):
        p = np.atleast_1d(np.asarray(p, dtype=float))
        x = np.full_like(p, np.nan)

        for i, piece in enumerate(self.pieces):
            in_piece = (p > self.cdf_thresholds[i]) & (p <= self.cdf_thresholds[i + 1])
            p_local = (p[in_piece] - self.cdf_thresholds[i]) / self.weights[i]
            x[in_piece] = piece.ppf(p_local)

        return x