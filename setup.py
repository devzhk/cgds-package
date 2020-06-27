import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CGDs-pkg",
    version="0.0.1",
    author="Hongkai Zheng",
    author_email="devzhk@gmail.com",
    description="A class of Pytorch optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devzhk/cgds-package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)