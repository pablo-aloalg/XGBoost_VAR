import numpy as np
from scipy.stats import genpareto

class GPDPiece:
    def __init__(self):
        pass

    def fit(self, data, lower=None, upper=None):
        data = np.asarray(data, dtype=float)
        self.x = data[~np.isnan(data)]          # quita los NaN
        self.n = len(self.x)

        if lower is None:
            raise ValueError("GPDPiece.fit: 'lower' threshold must be provided.")
        if self.n == 0:
            raise ValueError("GPDPiece.fit: no hay datos válidos (todo NaN o vacío).")
        if np.any(self.x <= lower):
            raise ValueError("GPDPiece.fit: all data must be above the threshold.")
        
        self.c, self.loc, self.scale = genpareto.fit(self.x, floc=lower)

        return self

    def pit_in_sample(self):
        # Calculate the PIT values for the in-sample data
        p = self.cdf(self.x)
        return p

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        p = genpareto.cdf(x, self.c, loc=self.loc, scale=self.scale)
        return p

    def ppf(self, p):
        p = np.asarray(p, dtype=float)
        x = genpareto.ppf(p, self.c, loc=self.loc, scale=self.scale)
        return x