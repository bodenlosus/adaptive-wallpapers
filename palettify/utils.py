import numpy as np
from palettify.types import RGBArray
from numba import types

def hexToRgbTuple(hex_code: types.unicode_type) -> RGBArray:
    hex_code = hex_code.lstrip('#')
    rgbs = np.zeros(shape=(3,), dtype=np.uint8)
    for i in range(3):
        rgbs[i] = int(hex_code[i * 2:i * 2 + 2], base=16)
    return rgbs