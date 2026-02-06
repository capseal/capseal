#!/usr/bin/env python3
"""Setup script for Fusion Alpha Python package"""

from setuptools import setup, find_packages
import os

# Read README if available
readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="fusion-alpha",
    version="0.1.0", 
    description="Committor planning for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fusion Alpha Team",
    author_email="contact@fusionalpha.dev",
    url="https://github.com/your-org/fusion-alpha",
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
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "benchmark": [
            "matplotlib>=3.0",
            "seaborn>=0.11",
            "pandas>=1.3",
        ],
        "ogbench": [
            "gym>=0.21",
            "mujoco-py>=2.1",
            "ogbench>=0.1",
        ],
    },
    package_data={
        "fusion_alpha": ["*.so", "*.pyd", "*.dll"],  # Include compiled binaries
    },
    include_package_data=True,
    zip_safe=False,
)