from setuptools import setup, find_packages

from my_pip_package import __version__

setup(
    name='NeuroMotor',
    version='dev',
    url='https://github.com/nemesgyadam/PyTorch-NeuroMotor',
    author='Nemes',
    author_email='nemesgyadam@gmail.com',

    py_modules=['my_pip_package'],
)
