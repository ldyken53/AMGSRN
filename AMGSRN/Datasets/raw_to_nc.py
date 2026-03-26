from pathlib import Path
from AMGSRN.Other.utility_functions import npy_to_cdf
import numpy as np

def raw_to_nc(raw_path: Path, dtype, shape, nc_path: Path):
    raw_data = np.fromfile(raw_path, dtype=dtype).reshape(shape, order='C')
    print(raw_data.min(), raw_data.max())
    # npy_to_cdf(raw_data[None,None], nc_path)
    # quit()
    raw_data = raw_data.astype(np.float32)
    raw_data -= raw_data.min()
    raw_data /= raw_data.max()
    print(raw_data.min(), raw_data.max())
    npy_to_cdf(raw_data[None,None], nc_path)

raw_to_nc(Path("../gaussian-volume/vtk/richtmyer_meshkov_2048x2048x1920_uint8.raw").resolve(), np.uint8, (1920, 2048, 2048), 
          Path("./Data/rm.nc").resolve())
