"""
DeepClean — Adaptive Dual-Mode Underwater Trash Detection System
Install with:  pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="deepclean-adaptive",
    version="1.0.0",
    author="Amol Bhatia, Krisvarish V., Harsh Yadav, Avishkar Jaiswal, Shivani Gupta, Saurav Gupta",
    description=(
        "Adaptive Dual-Mode Underwater Trash Detection System with "
        "3-Layer Intelligent Switching Controller"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/DeepClean-AdaptiveDualMode",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.3.0",
        "opencv-python>=4.9.0.80",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-image>=0.22.0",
        "matplotlib>=3.8.0",
        "pandas>=2.1.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "loguru>=0.7.0",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "datasets": ["roboflow>=1.1.0", "pycocotools>=2.0.7", "gdown>=5.1.0"],
    },
    entry_points={
        "console_scripts": [
            "deepclean-demo=run_adaptive_test:main",
            "deepclean-video=generate_synthetic_video:generate",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
