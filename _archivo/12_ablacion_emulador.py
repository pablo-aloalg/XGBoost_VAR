"""Ablación del emulador final (kNN + XGB + ruido): qué piezas se pueden quitar sin empeorar."""
import sys, pickle, json, time
sys.path.insert(0, "/home/claude/run")
import numpy as np, pandas as pd
from scipy.signal import lfilter, find_peaks
from scipy.stats import genextreme
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from utils.pit.piece_wise_marginal import from_normal
from utils.tv_var.harmonic_standardizer import HarmonicStandardizer

R = "/home/claude/run/"
waves = pd.read_csv(R + "inputs/waves_bulk.csv", index_col=0, parse_dates=True)
tr_ = pickle.load(open(R + "outputs/marginal/transformacion.pkl", "rb")); marg, harm_z = tr_["marg"], tr_["harm_z"]
def hs_desde_z(z):
    zz = pd.DataFrame({"VHM0": z, "VTM02": 0.0}, index=z.index)
    z0 = harm_z.inverse_transform(zz)["VHM0"].clip(-6.36, 6.36)
    return pd.Series(marg["VHM0"].ppf(from_normal(z0).values), index=z.index)

z_h = pd.read_csv(R + "outputs/marginal/z_bulk.csv", index_col=0, parse_dates=True)["VHM0"]
z6 = pd.read_csv(R + "outputs/marginal/z_bulk.csv", index_col=0, parse_dates=True).rolling("6h", center=True).mean()
z6 = z6[z6.index.hour.isin([0, 6, 12, 18])]["VHM0"].dropna()
hs_obs = waves.loc["1985":"2025", "VHM0"]

#### PCs 6-horarias: 30 seleccionadas y las 63
pcs30 = pd.read_csv(R + "outputs/seleccion_6h/pcs_seleccionadas_6h.csv", index_col=0, parse_dates=True)
allh = pd.read_csv(R + "outputs/pca/pcs_msl.csv", index_col=0, parse_dates=True).join(
       pd.read_csv(R + "outputs/pca/pcs_msl_gradient.csv", index_col=0, parse_dates=True))
pcs63 = allh[allh.index.hour.isin([0, 6, 12, 18])].reindex(pcs30.index)
LAGS = [0, 6, 12, 18, 24, 30, 36]
PAR = dict(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=200, subsample=0.8,
           colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=0)

def lagged(pcs):
    X = pd.concat({f"l{l}": pcs.shift(freq=pd.Timedelta(hours=l)) for l in LAGS}, axis=1).reindex(pcs.index)
    X.columns = [f"{c}_l{l[1:]}" for l, c in X.columns]
    return X.dropna()

def ajustar_xgb(pcs):
    """OOF (GroupKFold por año) en 1985–2015 + modelo final, como en 04B."""
    X = lagged(pcs); idx = X.index.intersection(z6.index); X, y = X.loc[idx], z6.loc[idx]
    tr = idx < "2016-01-01"
    f_oof = pd.Series(np.nan, index=X.index[tr])
    for a, b in GroupKFold(6).split(X[tr], y[tr], groups=X.index[tr].year):
        f_oof.iloc[b] = XGBRegressor(**PAR).fit(X[tr].iloc[a], y[tr].iloc[a]).predict(X[tr].iloc[b])
    m = XGBRegressor(**PAR).fit(X[tr], y[tr])
    return f_oof, m, list(X.columns)

t = time.time()
f_oof30, m30, c30 = ajustar_xgb(pcs30); print(f"XGB 30 PCs ({time.time()-t:.0f} s)", flush=True)
t = time.time()
f_oof63, m63, c63 = ajustar_xgb(pcs63); print(f"XGB 63 PCs ({time.time()-t:.0f} s)", flush=True)
XGB = {30: (f_oof30, m30, c30, pcs30), 63: (f_oof63, m63, c63, pcs63)}

def a_horaria(f6, suav):
    f = f6.resample("h").interpolate()
    return f.rolling(suav, center=True, min_periods=1).mean() if suav > 1 else f

#### Generador kNN (B = 12) sobre las 30 PCs: índices de origen, compartidos por todas las configuraciones
hist = pcs30.loc["1941":"2025"]; harm = HarmonicStandardizer(n_harmonics=3).fit(pcs30)
Zd = harm.transform(hist)[[f"msl_PC{i}" for i in range(1, 7)] + ["grad_PC2", "grad_PC3", "grad_PC4", "grad_PC5"]].values
doy = hist.index.dayofyear.values; n_h = len(hist); B, M = 12, 2
validos = np.arange(M - 1, n_h - B); cand = {}
for d in range(1, 367):
    dd = np.minimum(np.abs(doy[validos] - d), 366 - np.abs(doy[validos] - d)); cand[d] = validos[dd <= 15]
k_vec = int(np.sqrt(np.median([len(v) for v in cand.values()]))); pesos = 1 / np.arange(1, k_vec + 1); pesos /= pesos.sum()
def fuente_knn(n_anos, start, rng):
    fechas = pd.date_range(start, periods=n_anos * 1461, freq="6h"); n = len(fechas)
    fu = np.empty(n, dtype=int); j = rng.choice(cand[fechas[M - 1].day_of_year]); fu[:M] = np.arange(j - M + 1, j + 1); t = M
    while t < n:
        est = Zd[fu[t - M:t]].ravel(); c_ = cand[fechas[t].day_of_year]; c_ = c_[np.abs(c_ - fu[t - 1]) > 12]
        D = np.stack([Zd[c_ - M + 1 + m] for m in range(M)], axis=1).reshape(len(c_), -1)
        d2 = ((D - est) ** 2).sum(1); idx = np.argpartition(d2, k_vec)[:k_vec]
        i = rng.choice(c_[idx[np.argsort(d2[idx])]], p=pesos); m = min(B, n - t)
        fu[t:t + m] = np.arange(i + 1, i + 1 + m); t += m
    return fechas, fu

N_SIMS, N_ANOS = 6, 87
t = time.time()
FUENTES = [fuente_knn(N_ANOS, "1940-01-01", np.random.default_rng(i)) for i in range(N_SIMS)]
print(f"kNN: {N_SIMS} × {N_ANOS} años ({time.time()-t:.0f} s)", flush=True)

def f_sim(npcs, fechas, fu, suav):
    _, m, cols, pcs = XGB[npcs]
    p = pd.DataFrame(pcs.loc["1941":"2025"].values[fu], index=fechas, columns=pcs.columns)
    X = lagged(p)[cols]
    return a_horaria(pd.Series(m.predict(X), index=X.index), suav).loc["1941":]

def modelo_ruido(npcs, suav, escala, p):
    """Ajusta el ruido en 1985–2015 con la f OOF de esa configuración."""
    f = a_horaria(XGB[npcs][0], suav).loc["1985":"2015"]
    r = (z_h.reindex(f.index) - f)
    bordes = np.quantile(f.dropna(), np.linspace(0, 1, 11)[1:-1]); dec = np.digitize(f, bordes)
    ok = r.notna().values
    sigma = pd.Series(r.values[ok]).groupby(dec[ok]).std().reindex(range(10)).values
    u = r / sigma[dec] if escala else r.copy()
    X = np.column_stack([np.ones(len(u))] + [u.shift(k).values for k in range(1, p + 1)])
    okx = ~np.isnan(X).any(1) & ~np.isnan(u.values)
    beta, *_ = np.linalg.lstsq(X[okx], u.values[okx], rcond=None)
    e = np.full(len(u), np.nan); e[okx] = u.values[okx] - X[okx] @ beta
    pools = {k: np.sort(r.values[ok & (dec == k)]) for k in range(10)}
    return dict(bordes=bordes, sigma=sigma, c=beta[0], phi=beta[1:], e=e, pools=pools, escala=escala)

def ruido(f, mr, L, rng):
    n = len(f); ev = mr["e"]
    if L == 1:
        ee = ev[~np.isnan(ev)]; x = ee[rng.integers(len(ee), size=n)]
    else:
        ini = np.flatnonzero(np.convolve(~np.isnan(ev), np.ones(L), "valid") == L)
        s = rng.choice(ini, size=n // L + 1); x = ev[(s[:, None] + np.arange(L)).ravel()[:n]]
    burn = 24 * 30
    u = lfilter([1.0], np.r_[1, -mr["phi"]], np.r_[np.zeros(burn), x + mr["c"]])[burn:]
    dec = np.digitize(f, mr["bordes"])
    r = u * mr["sigma"][dec] if mr["escala"] else u
    out = np.empty(n)
    for k in range(10):                                        # mapeo de cuantiles por decil
        m = dec == k
        rk = (pd.Series(r[m]).rank().values - 0.5) / m.sum()
        out[m] = np.quantile(mr["pools"][k], rk)
    return pd.Series(out, index=f.index)

u_thr = hs_obs.quantile(0.99)
am_obs = hs_obs.groupby(hs_obs.index.year).max()
def metricas(am_f, am_hs, temp):
    nb = 41; allam = am_hs.values.ravel()
    bl = [am_hs[c].values[j:j + nb] for c in am_hs.columns for j in range(0, len(am_hs) - nb + 1, nb)]
    g = [genextreme.fit(b) for b in bl]
    lev = lambda T: np.median([genextreme.ppf(1 - 1 / T, *gg) for gg in g])
    return {"máx f": am_f.values.mean(), "temp/10a": np.median(temp), "Hs2": lev(2), "Hs10": lev(10), "Hs50": lev(50),
            "Hs100 emp": np.quantile(allam, 0.99), "std AM": np.median(am_hs.std())}

CONFIGS = {
    "A base (10)":                 dict(npcs=30, suav=13, escala=True,  L=6, p=48),
    "B sin σ(f)":                  dict(npcs=30, suav=13, escala=False, L=6, p=48),
    "C sin suavizado":             dict(npcs=30, suav=1,  escala=True,  L=6, p=48),
    "D sin bloques (L=1)":         dict(npcs=30, suav=13, escala=True,  L=1, p=48),
    "E p=24":                      dict(npcs=30, suav=13, escala=True,  L=6, p=24),
    "F 63 PCs (sin selección)":    dict(npcs=63, suav=13, escala=True,  L=6, p=48),
    "G todo simplificado":         dict(npcs=63, suav=1,  escala=False, L=1, p=24),
}
filas = {}
for nombre, cf in CONFIGS.items():
    t = time.time()
    mr = modelo_ruido(cf["npcs"], cf["suav"], cf["escala"], cf["p"])
    amf, amh, temp = [], [], []
    for i, (fechas, fu) in enumerate(FUENTES):
        f = f_sim(cf["npcs"], fechas, fu, cf["suav"])
        hs = hs_desde_z(f + ruido(f, mr, cf["L"], np.random.default_rng(100 + i)))
        amf.append(f.groupby(f.index.year).max().reset_index(drop=True))
        amh.append(hs.groupby(hs.index.year).max().reset_index(drop=True))
        temp.append(len(find_peaks(hs.values, height=u_thr, distance=72)[0]) * 10 / hs.index.year.nunique())
    filas[nombre] = metricas(pd.concat(amf, axis=1), pd.concat(amh, axis=1), temp)
    print(f"{nombre}: {({k: round(v, 2) for k, v in filas[nombre].items()})} ({time.time()-t:.0f} s)", flush=True)

obs_T = {"máx f": 2.62, "temp/10a": len(find_peaks(hs_obs.values, height=u_thr, distance=72)[0]) * 10 / 41,
         "Hs2": genextreme.ppf(0.5, *genextreme.fit(am_obs)), "Hs10": genextreme.ppf(0.9, *genextreme.fit(am_obs)),
         "Hs50": genextreme.ppf(0.98, *genextreme.fit(am_obs)), "Hs100 emp": np.nan, "std AM": am_obs.std()}
tabla = pd.DataFrame({"observado": obs_T, **filas}).T
print(tabla.round(2).to_string())
tabla.to_csv("/home/claude/abl/ablacion.csv")
