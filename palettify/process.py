import pathlib
from queue import Queue
from PIL import Image
import numpy as np
from palettify.conversion import applyPalette
from numpy.typing import NDArray
import curses
import threading

from palettify.palette import genPalette

lock = threading.Lock()

def singleFile(imagePath:str, outputPath:str, palette: NDArray):
    imagePath = pathlib.Path(imagePath)
    # Load the image and apply the palette
    image = Image.open(imagePath)
    arr: NDArray[np.float32] = np.array(image.convert('RGB'))
    
    # paletteImage(palette, stripeHeight=10)
    # Apply the palette mapping
    result_array: NDArray = applyPalette(arr, palette.astype(np.float32))

    # Convert back to an image and display
    palette_image = Image.fromarray(result_array)

    outputPath = pathlib.Path(outputPath)
    palette_image.save(outputPath)
    
def dirFiles(inputFolder: str, outputFolder: str, palettePath: str, batchProcess: bool = True, threadCount: int = 4):
    def wrapped(stdscr):
        inputFolderPath = pathlib.Path(inputFolder)
        outputFolderPath = pathlib.Path(outputFolder)
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        palette = genPalette(palettePath=palettePath)

        if batchProcess:
            fileQ = fQ()
            fileQ.startQueue(stdscr, threadCount)

        for image in inputFolderPath.iterdir():
            if not image.is_file():
                out(stdscr, 0, f"Skipping {image} because it is not a regular file.")
                continue
            if image.suffix.lower() in image_extensions:
                outputFile = outputFolderPath.joinpath(f"{image.stem}.o.png")
                if batchProcess:
                    fileQ.addJob(imagePath=image, outputPath=outputFile, palette=palette)
                else:
                    singleFile(imagePath=image, outputPath=outputFile, palette=palette)

        if batchProcess:
            fileQ.joinQueue()

    curses.wrapper(wrapped)

def worker(stdscr, fileQueue:Queue, id: int):
    while True:
        args = fileQueue.get()
        
        if not args:
            break
        
        count, imagePath, outputPath, palette = args
        
        out(stdscr, id, f"[{count}] Processing {imagePath.name}...")
        singleFile(imagePath, outputPath, palette)
        out(stdscr, id, f"[{count}] Finished {imagePath.name}!")
        
        fileQueue.task_done()

class fQ():
    def __init__(self,):
        self.q = Queue()
        self.threads = []
        self.threadCount = 0
        self.jobNumber = 0
    
    def startQueue(self, stdscr, threadCount:int=4):
        for id in range(threadCount):
            t = threading.Thread(target=worker, args=(stdscr, self.q, id))
            t.start()
            out(stdscr, id, f"Worker {id} started!")
            
            self.threadCount += 1
            self.threads.append(t)
        
        stdscr.getch()
    
    def addJob(self, imagePath: str, outputPath: str, palette: NDArray):
        self.jobNumber += 1
        self.q.put((self.jobNumber, imagePath, outputPath, palette))
    
    def joinQueue(self):
        self.q.join()
        
        for _ in range(self.threadCount):
            self.q.put(None)
        
        for t in self.threads:
            t.join()

def out(stdscr, id, msg):
    global lock
    with lock:
        height, width = stdscr.getmaxyx()
        stdscr.addstr(id, 0, msg[:width - 1])  # Truncate to fit terminal width
        stdscr.refresh()