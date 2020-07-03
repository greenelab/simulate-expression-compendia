# First manually 
# conda activate simulate_expression_compendia
# pip install git+https://github.com/ajlee21/ponyo.git@pypi
pytest -v --nbval-lax --current-env Pseudomonas_tests/*.ipynb
pytest -v --nbval-lax --current-env Human_tests/*.ipynb