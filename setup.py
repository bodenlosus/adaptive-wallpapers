from setuptools import setup, find_packages

setup(name='palettify',
      version='1.0',
      # Modules to import from other scripts:
      packages=find_packages(),
      # Executables
      entry_points={
        'console_scripts': [
            'palettify=palettify.cli:cli',
        ]
    }
     )