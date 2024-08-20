from setuptools import setup, find_packages
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
#with open (os.path.join(here, "spared", "README.md"), "r") as f:
#   long_description = f.read()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="spared",
    version="1.0.10",
    author="Daniela Vega",
    author_email="d.vegaa@uniandes.edu.co",
    description="SpaRED and Spackle library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dvegaa00/Library_Spared_Spackle/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "spared"},
    packages=find_packages(where="spared"),
    include_package_data=True,
    python_requires=">=3.7",
)

