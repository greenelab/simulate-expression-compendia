# Correcting for experiment-specific variability in expression compendia can remove underlying signals

**Alexandra J Lee, YoSon Park, Georgia Doing, Deborah A Hogan and Casey S Greene**

**University of Pennsylvania, Dartmouth College**

[![PDF Manuscript](https://img.shields.io/badge/manuscript-PDF-blue.svg)](https://academic.oup.com/gigascience/article/9/11/giaa117/5952607)

This repository stores data and analysis modules to simulate compendia of gene expression data and measure the effect of technical sources of variation on our ability to extract an underlying biological signal.  

**Motivation:** In the last two decades, scientists working in different labs have assayed gene expression from millions of samples. These experiments can be combined into a compendium and used to extract novel biological patterns. However, combining different experiments introduces technical variance, which could distort biological patterns and lead to misinterpretation. As the scale and prevalence of these compendia increases, it becomes crucial to evaluate how integrating multiple experiments affects our ability to detect biological patterns.

**Objective:** To determine the extent to which underlying biological structures are masked by technical variants via simulation of a multi-experiment compendia.

**Method:** We used a generative multi-layer neural network to simulate a compendium of P. aeruginosa gene expression experiments. We performed a pairwise comparison of the simulated compendium versus the simulated compendium containing varying number of sources of technical variation.

**Results:** We found that it was difficult to detect the original biological structure of interest in a compendium containing some sources of technical variation unless we applied batch correction. Interestingly, as the number of sources of variation increased, it became easier to detect the original biological structure without correction. Furthermore, when we applied batch correction, it reduced our power to detect the biological structure of interest.     

**Conclusion:** When combining some sources of technical variation, it is best to perform batch correction. However, as the number of sources increases, batch correction becomes unnecessary and indeed harms our ability to extract biological patterns.

**Citation:**
For more details about the analysis, see our paper published in GigaScience. The paper should be cited as:
> Alexandra J Lee, YoSon Park, Georgia Doing, Deborah A Hogan, Casey S Greene, Correcting for experiment-specific variability in expression compendia can remove underlying signals, GigaScience, Volume 9, Issue 11, November 2020, giaa117, https://doi.org/10.1093/gigascience/giaa117

## Analysis Modules

There are 2 analyses using Pseudomonas dataset in the `Pseudomonas` directory and 2 analyses using the recount2 dataset in the `Human` directory:

| Name | Description |
| :--- | :---------- |
| [Pseudomonas_sample_lvl_sim](Pseudomonas/Pseudomonas_sample_lvl_sim.ipynb) | Analysis notebook applying sample-level gene expression simulation to *P. aeruginosa* data|
| [Pseudomonas_experiment_lvl_sim](Pseudomonas/Pseudomonas_experiment_lvl_sim.ipynb) | Analysis notebook applying experiment-level gene expression simulation to *P. aeruginosa* data|
| [Human_sample_lvl_sim](Human/Human_sample_lvl_sim.ipynb) | Analysis notebook applying sample-level gene expression simulation to human (recount2) data|
| [Human_experiment_lvl_sim](Human/Human_experiment_lvl_sim.ipynb) | Analysis notebook applying experiment-level gene expression simulation to human (recount2) data|


## Usage

**How to run notebooks from simulate-expression-compendia**

*Operating Systems:* Mac OS, Linux

In order to run this simulation on your own gene expression data the following steps should be performed:

First you need to set up your local repository: 
1. Download and install [github's large file tracker](https://git-lfs.github.com/).
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Clone the `simulate-expression-compendia` repository by running the following command in the terminal:
```
git clone https://github.com/greenelab/simulate-expression-compendia.git
```
Note: Git automatically detects the LFS-tracked files and clones them via http.
4. Navigate into cloned repo by running the following command in the terminal:
```
cd simulate-expression-compendia
```
5. Set up conda environment by running the following command in the terminal:
```bash
# conda version 4.6.12
conda env create -f environment.yml

conda activate simulate_expression_compendia

pip install -e .
```
6. Navigate to either the `Pseudomonas` or `Human` directories and run the notebooks.


**How to analyze your own data**

In order to run this simulation on your own gene expression data the following steps should be performed:

First you need to set up your local repository and environment: 
1. Download and install [github's large file tracker](https://git-lfs.github.com/).
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Clone the `simulate-expression-compendia` repository by running the following command in the terminal:
```
git clone https://github.com/greenelab/simulate-expression-compendia.git
```
Note: Git automatically detects the LFS-tracked files and clones them via http.
4. Navigate into cloned repo by running the following command in the terminal:
```
cd simulate-expression-compendia
```
5. Set up conda environment by running the following command in the terminal:
```bash
# conda version 4.6.12
conda env create -f environment.yml

conda activate simulate_expression_compendia

pip install -e .
```
6. Create a new analysis folder in the main directory. This is equivalent to the `Pseudomonas` directory
7. Copy `Pseudomonas_sample_lvl_sim.ipynb` or `Pseudomonas_experiment_lvl_sim.ipynb` into your analysis folder depending on if you would like to use the sample level(see [simulate_by_random_sampling()](https://github.com/greenelab/ponyo/blob/master/ponyo/simulate_expression_data.py)) or experiment level simulation (see [simulate_by_latent_transformation()](https://github.com/greenelab/ponyo/blob/master/ponyo/simulate_expression_data.py))approach. 
8. Within your analysis folder create `data/` directory and `input/`, `metadata/` subdirectories

Next we need to modify the code for your analysis:
1. Create a configuration file in `configs/` using the parameters outlined below.
2. Update the analysis notebooks to use your config file (see below) and input file
3. Add your gene expression data file to the `data/input/` directory.  Your data is expected to be stored as a tab-delimited dataset with samples as rows and genes as columns. Your input data is also expected to be 0-1 normalized per gene. If your data needs to be normalized or transposed, there are functions to do this in [ponyo/utils](https://github.com/greenelab/ponyo/blob/master/ponyo/utils.py).
4. Add your metadata file to `data/metadata/` directory.  Your metadata is expected to be stored as a tab-delimited with sample ids matching the gene expression dataset as one column and experiment ids as another. 
5. Run notebooks

**Additional customization**

Further customization can be accomplished by doing the following:

1. The `apply_correction_io` function in the `generate_data_parallel.py` file can be modified to use a different correction method.
2. If there are additional pre-processing specific to your data, these can be added as modules in the `pipeline.py` file and called in the analysis notebook

**Configuration file**

The tables lists parameters required to run the analysis in this repository.

Note: Some of these parameters are required by the imported [ponyo](https://github.com/greenelab/ponyo) modules. 

| Name | Description |
| :--- | :---------- |
| local_dir| str: Parent directory on local machine to store intermediate results.|
| scaler_transform_file| str: File name to store mapping from normalized to raw gene expression range. This is an intermediate file that gets generated. This file is generated in the `normalize_expression_data()` function from this [ponyo script](https://github.com/greenelab/ponyo/blob/master/ponyo/train_vae_modules.py).|
| dataset_name| str: Name for analysis directory. Either "Human" or "Pseudomonas". If you created a new analysis directory this is the name of that new directory created in step 6 above.|
| simulation_type | str: "sample_lvl_sim" (simulated based on randomly sampling the latent space) or "experiment_lvl_sim" (simulation based on shifting in the latent space).|
| NN_architecture | str: Name of neural network architecture to use. Format 'NN_<intermediate layer>_<latent layer>'.|
| learning_rate| float: Step size used for gradient descent. In other words, it's how quickly the  methods is learning.|
| batch_size | str: Training is performed in batches. So this determines the number of samples to consider at a given time.|
| epochs | int: Number of times to train over the entire input dataset.|
| kappa | float: How fast to linearly ramp up KL loss.|
| intermediate_dim| int: Size of the hidden layer.|
| latent_dim | int: Size of the bottleneck layer.|
| epsilon_std | float: Standard deviation of Normal distribution to sample latent space.|
| validation_frac | float: Fraction of input samples to use to validate for VAE training.|
| num_simulated_samples | int: Simulate a compendium with this number of samples. Used if simulation_type == "sample_lvl_sample"|
| num_simulated_experiments| int: Simulate a compendium with this number of experiments. Used if simulation_type == "experiment_lvl_sample"|
| lst_num_experiments | list:  List of different numbers of experiments to add to simulated compendium.  These are the number of sources of technical variation that are added to the simulated compendium.|
| lst_num_partitions | list:  List of different numbers of partitions to add to simulated compendium.  These are the number of sources of technical variation that are added to the simulated compendium.|
| use_pca | bool: True if want to represent expression data in top PCs before calculating SVCCA similarity.|
| num_PCs | int: Number of top PCs to use to represent expression data. If use_pca == True.|
| correction_method | str: Noise correction method to use. Either "limma" or "combat".|
| metadata_colname | str: Column header that contains sample id that maps expression data and metadata.|
| iterations | int: Number of simulations to run.|
| num_cores | int: Number of processing cores to use.|

## Acknowledgements
We would like to thank YoSon Park, David Nicholson, Ben Heil and Ariel Hippen-Anderson for insightful discussions and code review
