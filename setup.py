import os

import setuptools

readme_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
with open(readme_filepath, "r") as fh:
    long_description = fh.read()

version_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
with open(version_filepath, "r") as fh:
    version = fh.read().strip()

setuptools.setup(
    name="smoe",
    version=version,
    author="MoE Group",
    author_email="tzhu1997@outlook.com",
    description="A toolkit for LLM MoE and continual pretraining.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/Spico197/smoe",
    packages=setuptools.find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3.0",
        "omegaconf>=2.0.6",
        "tqdm>=4.65.0",
        "datasets>=2.13.1",
        "transformers>=4.30.2",
        "peft>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "coverage",
            "black",
            "isort",
            "flake8",
            "pre-commit",
        ]
    },
    include_package_data=True,
    entry_points={},
)
