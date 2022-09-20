from setuptools import setup, Extension
import pybind11
import jaxlib

setup(
    packages = ["add_one"],
    ext_modules = [
      Extension(
        name="add_one.add_one_lib",
        sources=["lib/add_one_lib.cc"],
        include_dirs = [
          pybind11.get_include(), 
          jaxlib.get_include()], 
        )])
