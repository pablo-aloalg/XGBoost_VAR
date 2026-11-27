"""ERA5 msl horario 0.25° (netcdf por año, cluster) → zarr 6 h en el dominio WMed, con msl y grad2.

Misma estructura que data/era5_mslp_grad_1985_2026_1deg_hourly.zarr (valid_time, latitude, longitude;
msl y grad2 en float32), así que el 01 y el 03 solo cambian la ruta.
El gradiente se calcula con 1° de margen y luego se recorta: spatial_gradient pone los bordes a 0
y así esos bordes caen fuera del dominio, como pasaba en el de 1°.
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'                # Lustre

import numpy as np
import xarray as xr
from bluemath_tk.core.operations import spatial_gradient

SRC = '/lustre/geocean/WORK/users/alonsoap/projects/TWINPEACS/02_Emulator_Testing/daily_atmospheric/outputs/data/era5_hourly/era5_mslp_{}_025deg.nc'      # <- tu ruta
DST = 'era5_mslp_grad_1940_2026_025deg_6h.zarr'
YEARS = range(1940, 2027)

LAT_M, LON_M = slice(49, 29), slice(-6, 21)                 # dominio + 1° de margen (lat descendente)
LAT, LON = slice(48, 30), slice(-5, 20)                     # dominio WMed (01/03)

for year in YEARS:
    ds = xr.open_dataset(SRC.format(year), engine='h5netcdf', chunks={'valid_time': 744})
    ds = ds[['msl']].drop_vars(['number', 'expver'], errors='ignore')
    ds = ds.sel(latitude=LAT_M, longitude=LON_M)                 # recorte espacial primero (slice barato)
    ds = ds.isel(valid_time=slice(0, None, 6))                   # 00/06/12/18 si el fichero empieza a las 00
    assert (ds.valid_time.dt.hour % 6 == 0).all()
    ds = ds.load()

    grad2 = spatial_gradient(ds['msl'].rename({'valid_time': 'time'})).rename({'time': 'valid_time'})
    ds['grad2'] = grad2
    ds = ds.sel(latitude=LAT, longitude=LON).astype('float32')

    for v in ds.variables:
        ds[v].encoding = {}                                  # quita zlib/chunks del netcdf
    ds = ds.chunk({'valid_time': -1, 'latitude': -1, 'longitude': -1})

    if not os.path.exists(DST):
        ds.to_zarr(DST, mode='w')
    else:
        ds.to_zarr(DST, append_dim='valid_time')
    print(year, ds.sizes['valid_time'], flush=True)

out = xr.open_zarr(DST)
print(out)
print('huecos 6 h:', int((np.diff(out.valid_time.values) != np.timedelta64(6, 'h')).sum()))
