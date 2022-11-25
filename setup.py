from cx_Freeze import setup, Executable
from setuptools import find_packages
import sys

setup(
name='text_detector',

version='0.1',

executables = [Executable("eval.py")]
)