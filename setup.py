from setuptools import setup, find_packages

setup(
    name="qscaled",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "ipykernel",
        "matplotlib",
        "pandas",
        "rliable",
        "scikit-learn",
        "scipy",
        "seaborn",
        "tqdm",
        "wandb",
    ],
)
