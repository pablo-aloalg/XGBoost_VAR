import numpy as np
from scipy.stats import rankdata

class EmpiricalPiece:
    def __init__(self):
        pass

    def fit(self, data, lower=None, upper=None):

        data = np.asarray(data, dtype=float)
        self.x = data[~np.isnan(data)]          # quita los NaN
        self.n = len(self.x)
        if self.n == 0:
            raise ValueError("EmpiricalPiece.fit: no hay datos válidos (todo NaN o vacío).")
        
        self.p_in_sample = rankdata(self.x) / (self.n + 1)
        x_sorted = np.sort(self.x)
        p_sorted = rankdata(x_sorted) / (self.n + 1)

        self.x_table, idx = np.unique(x_sorted, return_index=True)
        self.p_table = p_sorted[idx]

        return self

    def pit_in_sample(self):
        return self.p_in_sample

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        p = np.interp(x, self.x_table, self.p_table)
        return p

    def ppf(self, p):
        p = np.asarray(p, dtype=float)
        x = np.interp(p, self.p_table, self.x_table)
        return x