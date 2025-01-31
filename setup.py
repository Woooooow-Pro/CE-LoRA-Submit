from setuptools import setup, find_packages

__version__ = "0.0.0"

setup(
    name="CELoRA",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
