from setuptools import setup, find_packages

setup(
    name="bca_survival_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "matplotlib", "lifelines", "scikit-learn"],
    entry_points={
        "console_scripts": [],
    },
)
