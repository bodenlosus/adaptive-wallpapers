from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from numpy.typing import NDArray
import numpy.typing as npt
from numba import njit, types, prange
import pathlib
import sys

PALETTE_SIZE: int = 16

# Type aliases for readability
RGBArray = types.Array(types.uint8, 1, 'C')  # 1D array for a single RGB color
PaletteArray = types.Array(types.uint8, 2, 'C')  # 2D array for the palette
ImageArray = types.Array(types.uint8, 3, 'C')  # 3D array for image data (H x W x RGB)


def hexToRgbTuple(hex_code: types.unicode_type) -> RGBArray:
    hex_code = hex_code.lstrip('#')
    rgbs = np.zeros(shape=(3,), dtype=np.uint8)
    for i in range(3):
        rgbs[i] = int(hex_code[i * 2:i * 2 + 2], base=16)
    return rgbs

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

def genPalette(palettePath: str, expandSize = 4):
    palettePath = pathlib.Path(palettePath)
    with open(palettePath, 'r') as f:
        hexCodes = f.readlines()
        palette: NDArray[np.uint8] = np.zeros(shape=(PALETTE_SIZE, 3), dtype=np.uint8)
        
        for i in range(PALETTE_SIZE):
            rgb = hexToRgbTuple(hexCodes[i].strip())
            palette[i] = rgb
        
        if expandSize > 0:
            return d2lerpPalette(palette)
        
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
    
    img.show()

@njit
def findClosestColor(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Find the closest color in the palette to the given RGB color."""
    # Calculate squared distances for all palette colors
    dists = np.sum((palette - rgb) ** 2, axis=1)
    # Find the index of the minimum distance
    min_index = np.argmin(dists)
    # Return the closest color
    return palette[min_index]

@njit
def applyPalette(arr: ImageArray, palette: PaletteArray) -> ImageArray:
    """Map the image array to the closest color in the palette."""
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = findClosestColor(arr[i, j], palette)
    return arr

def singleFile(imagePath:str, outputPath:str, expandSize: int, palettePath: str):
    imagePath = pathlib.Path(imagePath)
    # Load the image and apply the palette
    image = Image.open(imagePath)
    arr: NDArray[np.uint8] = np.array(image)

    palette = genPalette(palettePath=palettePath, expandSize=expandSize)
    
    # paletteImage(palette, stripeHeight=10)
    # Apply the palette mapping
    result_array: NDArray[np.uint8] = applyPalette(arr, palette)

    # Convert back to an image and display
    palette_image = Image.fromarray(result_array)
    outputPath = pathlib.Path(outputPath)
    palette_image.save(outputPath, format="PNG")

def dirFiles(inputFolder: str, outputFolder: str, expandSize: int, palettePath: str):
    inputFolder = pathlib.Path(inputFolder)
    outputFolder = pathlib.Path(outputFolder)
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    
    for image in inputFolder.iterdir():
        if image.suffix.lower() in image_extensions:
            singleFile(imagePath=image, outputPath=outputFolder.joinpath(f"{image}.o.png"), expandSize=expandSize, palettePath=palettePath)
        
    # Load the image and apply the palette
def main(imagePath:str, outputPath:str, expandSize: int, palettePath: str, dir: bool=False):
    if dir:
        dirFiles(inputFolder=imagePath, outputFolder=outputPath, expandSize=expandSize, palettePath=palettePath)
    else:
        singleFile(imagePath=imagePath, outputPath=outputPath, expandSize=expandSize, palettePath=palettePath)

if __name__ == "__main__":
    main(sys.argv[1])