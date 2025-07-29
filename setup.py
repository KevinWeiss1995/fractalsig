from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="fractalsig",
    version="0.1.0",
    description="Python library for generating and analyzing fractional Gaussian noise and related transforms",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
    author="fractalsig",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
) 