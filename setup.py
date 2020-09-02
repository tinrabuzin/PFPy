import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pfpy",
    version="0.0.2",
    author="Tin Rabuzin",
    author_email="trabuzin@gmail.com",
    description="Power Factory with Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tinrabuzin/PFPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'pandas',
        'numpy', 
        'matplotlib',
        'scipy'
    ]
)