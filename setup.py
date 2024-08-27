from setuptools import setup, find_packages

setup(
    name="timesfm_torch",
    version="0.1",
    packages=find_packages(),
    install_requires=[
    ],
    author="Hao Tang",
    author_email="tangh18.thu@gmail.com",
    description="",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tangh18/timesfm_torch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
