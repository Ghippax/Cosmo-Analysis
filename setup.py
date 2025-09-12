from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent

# Read long description for PyPI/GitHub
long_description = (this_dir / "README.md").read_text(encoding="utf-8") if (this_dir / "README.md").exists() else ""

# Read requirements.txt and filter comments/empty lines
req_file = this_dir / "requirements.txt"
if req_file.exists():
    requirements = [line.strip() for line in req_file.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")]
else:
    requirements = []

setup(
    name="cosmo_analysis",
    version="0.0.0",
    description="Analysis utilities for Gadget/AGORA simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pablo Granizo Cuadrado",
    author_email="pablogranizopgc@gmail.com",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=requirements,
    )
