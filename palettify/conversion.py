import numpy as np
from numba import njit
from palettify.types import *

@njit
def interpolate(color: np.ndarray, closest1: np.ndarray, closest2: np.ndarray) -> np.ndarray:
    d:np.float32 = np.sum(np.square(closest2 - closest1), dtype=np.float32)
    
    if d == 0:
        return closest1
    
    f = np.dot(color - closest1, closest2-closest1) / d
    if f <= 0:
        return closest1
    elif f >= 1:
        return closest2

    return closest1 + f * (closest2 - closest1)
    
@njit
def findClosestColors(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Find the closest color in the palette to the given RGB color."""
    # Calculate squared distances for all palette colors
    dists = np.sum((palette - rgb) ** 2, axis=1)
    # Find the index of the minimum distance
    min_index = np.argmin(dists)
    min_index2 = np.argmin(dists[dists != dists[min_index]])
    # Return the closest color
    color = interpolate(rgb, palette[min_index], palette[min_index2])
    return color

@njit
def applyPalette(arr: ImageArray, palette: PaletteArray) -> ImageArray:
    """Map the image array to the closest color in the palette."""
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            old = arr[i, j]
            new = findClosestColors(arr[i, j], palette)
            arr[i, j] = new
            
            error = old - new
            
            # Floyd steinberg dithering
            if i + 1 < arr.shape[0]:
                arr[i+1, j] = arr[i+1, j] + error * 7 / 16
            if i - 1 >= 0 and j + 1 < arr.shape[1]:
                arr[i-1, j+1] = arr[i-1, j+1] + error * 3 / 16
            if j + 1 < arr.shape[1]:
                arr[i, j+1] = arr[i, j+1] + error * 5 / 16
            if i + 1 < arr.shape[0] and j + 1 < arr.shape[1]:
                arr[i+1, j-1] = arr[i+1, j-1] + error * 1 / 16
                
            
    return arr


    # Load the image and apply the palette
