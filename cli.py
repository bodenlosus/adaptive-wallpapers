import argparse
from main import main
import sys
def cli():
    parser = argparse.ArgumentParser(
        prog='Adaptive Wallpapers',
        description='Convert your wallpapers to your favpurite pallete',
        epilog='')
    
    parser.add_argument('-i', '--input', required=True, type=str)
    parser.add_argument('-d', '--dir', required=False, action='store_true', default=False)
    parser.add_argument('-o', '--output', required=False, default='output.png', type=str)
    parser.add_argument('-p', '--palette', required=True, type=str)
    parser.add_argument('-ip', '--interpolate', required=False, default=0, type=int)
    
    args = parser.parse_args()
    
    main(
        imagePath=args.input,
        outputPath=args.output,
        palettePath=args.palette,
        expandSize=args.interpolate,
        dir=args.dir
        )

if __name__ == "__main__":
    cli()