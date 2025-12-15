"""Setup configuration for integrated_path_planning package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding='utf-8')

setup(
    name="integrated_path_planning",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Integrated path planning with pedestrian trajectory prediction",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/integrated_path_planning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "attrdict>=2.0.1",
        "numba>=0.57.0",
        "loguru>=0.7.0",
        "dataclasses-json>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "seaborn>=0.12.0",
            "pillow>=9.0.0",
        ],
    },
)
