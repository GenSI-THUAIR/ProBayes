from setuptools import setup, find_packages

setup(
    name="pepglad",
    version="0.0.1",
    packages=find_packages(),
    package_dir={
        'utils': './utils',
        'data':'./data'
    },
)
