import pathlib
from PIL import Image
import numpy as np
import numpy.typing as npt
from palettify.types import *
from palettify.utils import hexToRgbTuple
def lLerpPalette(palette: PaletteArray, expandSize: int = 3) -> PaletteArray:
    newPalette = []
    for i in range(palette.shape[0] - 1):
        for j in range(expandSize + 1):
            newColor = palette[i] * (j/expandSize) + palette[i+1] * (1 - j/expandSize)
            newPalette.append(newColor)
    
    newPalette.append(palette[-1])
    
    return np.array(newPalette, dtype=np.uint8)

def d2lerpPalette(palette: PaletteArray, lerpSize: int=2):
    palette = np.asarray(palette, dtype=np.float32)  # Ensure calculations are done in float
    n_colors = palette.shape[0]
    newPalette = [palette]

    for i in range(n_colors - 1):
        for j in range(i + 1, n_colors):
            # Compute linear interpolation steps for colors i and j
            t = np.linspace(0, 1, lerpSize + 1) # Exclude endpoints 0 and 1
            interpolated_colors = palette[i] * t[:, None] + palette[j] * (1 - t[:, None])
            newPalette.append(interpolated_colors)

    return np.vstack(newPalette).astype(np.uint8)

def genPalette(palettePath: str, PALETTE_SIZE=16):
    palettePath = pathlib.Path(palettePath)
    with open(palettePath, 'r') as f:
        hexCodes = f.readlines()
        palette: npt.NDArray[np.uint8] = np.zeros(shape=(PALETTE_SIZE, 3), dtype=np.uint8)
        
        for i in range(PALETTE_SIZE):
            rgb = hexToRgbTuple(hexCodes[i].strip())
            palette[i] = rgb
            
        palette = d2lerpPalette(palette)
        
        paletteImage(palette, 10).save('palette.png')
        return palette

def paletteImage(palette: PaletteArray, stripeHeight):
    width = palette.shape[0]
    height = stripeHeight * len(palette)
    img = Image.new('RGB', (height, width, ), (255, 255, 255))
    
    # Fill the image with broader stripes
    for i, color in enumerate(palette):
        for y in range(i * stripeHeight, (i + 1) * stripeHeight):
            for x in range(width):
                img.putpixel((y, x), tuple(color))  # Set pixel color
    
    return img