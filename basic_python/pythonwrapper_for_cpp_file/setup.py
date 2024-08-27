from setuptools import setup, Extension

sfc_module = Extension('superfastcode', sources = ['fastcpp_using_cpython.cpp'])

setup(
    name='superfastcode',
    version='1.0',
    description='Python Package with superfastcode C++ extension',
    ext_modules=[sfc_module]
)