from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="spared",
    version="2.0.3",
    author="Daniela Vega",
    author_email="d.vegaa@uniandes.edu.co",
    description="SpaRED and Spackle library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BCV-Uniandes/SpaRED/tree/main",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "spared"},
    packages=find_packages(where="spared"),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        # List any additional dependencies here if necessary
    ],
    extras_require={
        # Optional dependencies
    },
    project_urls={
        "Bug Reports": "https://github.com/BCV-Uniandes/SpaRED/issues",
        "Source": "https://github.com/BCV-Uniandes/SpaRED/",
    },
)

