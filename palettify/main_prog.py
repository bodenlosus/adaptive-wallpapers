from palettify.palette import genPalette
from palettify.process import dirFiles, singleFile


def main(imagePath:str, outputPath:str, palettePath: str, dir: bool=False):
    if dir:
        dirFiles(inputFolder=imagePath, outputFolder=outputPath, palettePath=palettePath)
    else:
        singleFile(imagePath=imagePath, outputPath=outputPath, palette=genPalette(palettePath))