import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepMetaProteome",
    version="0.0.1",
    author="Xiaofang Li",
    author_email="xfli@sjziam.ac.cn",
    description="A deep learning package for mining specific proteins from metaproteomes/metagenomes",
    long_description=long_description,
    long_description_content_type="this package include two moduals comprising functions for parsing FASTA dataset, preprocessing of training data and prediction dataset, building and training deep learning model and use of the model for prediction/markdown",
    url="https://github.com/uqxli12/DeepMetaProteome.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
