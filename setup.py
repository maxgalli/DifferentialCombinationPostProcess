import setuptools
from importlib import import_module

setuptools.setup(
    name="differential_combination_postprocess",
    author="Massimiliano Galli",
    author_email="massimiliano.galli.95@gmail.com",
    description="Package for Run 2 differential combination postprocessing",
    packages=setuptools.find_packages(),
    scripts=["scripts/plot_xs_scans.py", "scripts/quick_scan.py", "scripts/plot_TK_scans.py", "scripts/plot_SMEFT_scans.py", "scripts/plot_matrices.py", "scripts/plot_SMEFT_linquad.py", "scripts/plot_two_scans.py", "scripts/plot_scan_chisquare.py"],
    python_requires=">=3.6",
)
