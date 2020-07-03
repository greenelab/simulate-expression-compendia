import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="simulate_expression_compendia_modules",
    version="0.1",
    description="Install functions to run analysis to examine effects of technical variation in large-scale gene expression compendia",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/greenelab/simulate-expression-compendia",
    author="Alexandra Lee",
    author_email="alexjlee.21@gmail.com",
    license="BSD 3-Clause",
    packages=["simulate_expression_compendia_modules"],
    zip_safe=False,
)
