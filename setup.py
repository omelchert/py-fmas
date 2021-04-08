from setuptools import setup

setup(
    name = "py-fmas",
    version = "1.3.0",
    author = "Oliver Melchert",
    author_email = "melchert@iqo.uni-hannover.de",
    description = "Ultrashort optical pulse propagation in terms of the analytic signal",
    keywords = "Ultrashort pulse propagation, analytic signal, Raman effect, Spectrograms",
    license = "MIT",
    packages=['fmas'],
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        "numpy>=1.19.4",
        "scipy>=1.5.4",
        "matplotlib>=3.3.3",
        "h5py>=3.1.0"
    ],
    python_requires='>=3.9',
)
