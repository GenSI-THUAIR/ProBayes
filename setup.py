from setuptools import setup, find_packages

setup(
    name="probayes",
    version="0.0.0",
    packages=find_packages(),
    package_dir={
        'openfold': './openfold',
        'probayes': './probayes',
        'core': './probayes/core',
        'data': './data',
    },
)
