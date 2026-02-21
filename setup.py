from pathlib import Path
from setuptools import setup, find_packages

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")


setup(
    name="TabDDPM-Aug",
    version="1.0.0",
    description="Adaptive diffusion-based hybrid augmentation for imbalanced tabular classification",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/tanveerkalwar/tabddpm-aug",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=2.0.2",
        "pandas>=2.2.2",
        "scipy>=1.16.3",
        "scikit-learn>=1.6.1",
        "imbalanced-learn>=0.14.0",
        "torch>=2.9.0",
        "ctgan>=0.11.1",
        "sdv>=1.32.1",
        "catboost>=1.2.8",
        "xgboost>=3.1.2",
    ],
)
