# The impact of undesired technical variability on large-scale data compendia

**Alexandra J Lee, YoSon Park, Georgia Doing, Deborah A Hogan and Casey S Greene**

**January 2020**

**University of Pennsylvania**

This repository stores data and analysis modules to simulate compendia of gene expression data and measure the effect of technical sources of variation on our ability to extract an underlying biological signal.  

*Motivation:* In the last two decades, scientists working in different labs have assayed gene expression from millions of samples. These experiments can be combined into a compendium and used to extract novel biological patterns. However, combining different experiments introduces technical variance, which could distort biological patterns and lead to misinterpretation. As the scale and prevalence of these compendia increases, it becomes crucial to evaluate how integrating multiple experiments affects our ability to detect biological patterns.

*Objective:* To determine the extent to which underlying biological structures are masked by technical variants via simulation of a multi-experiment compendia.

*Method:* We used a generative multi-layer neural network to simulate a compendium of P. aeruginosa gene expression experiments. We performed a pairwise comparison of the simulated compendium versus the simulated compendium containing varying number of sources of technical variation.

*Results:* We found that it was difficult to detect the original biological structure of interest in a compendium containing some sources of technical variation unless we applied batch correction. Interestingly, as the number of sources of variation increased, it became easier to detect the original biological structure without correction. Furthermore, when we applied batch correction, it reduced our power to detect the biological structure of interest.     

*Conclusion:* When combining some sources of technical variation, it is best to perform batch correction. However, as the number of sources increases, batch correction becomes unnecessary and indeed harms our ability to extract biological patterns.

## Analysis Modules

There are 2 analyses using Pseudomonas dataset in the `Pseudomonas` directory and 2 analyses using the recount2 dataset in the `Human` directory:

| Name | Description |
| :--- | :---------- |
| [Pseudomonas_sample_lvl_sim](Pseudomonas/Pseudomonas_sample_lvl_sim.ipynb) | Pseudomonas sample-level gene expression simulation|
| [Pseudomonas_experiment_lvl_sim](Pseudomonas/Pseudomonas_experiment_lvl_sim.ipynb) | Pseudomonas experiment-level gene expression simulation|
| [Human_sample_lvl_sim](Human/Human_sample_lvl_sim.ipynb) | Human sample-level gene expression simulation|
| [Human_experiment_lvl_sim](Human/Human_experiment_lvl_sim.ipynb) | Human experiment-level gene expression simulation|


## Computational Environment

All processing and analysis scripts were performed using the conda environment specified in `environment.yml`.
To build and activate this environment run:

```bash
# conda version 4.6.12
conda env create -f environment.yml

conda activate simulate_expression_compendia
```

## How to run this simulation using your own data

In order to run this simulation on your own gene expression data the following steps should be performed:

First we need to set up your local repository: 
1. Clone the ```simulate-expression-compendia``` repository
2. Set up conda environment using the command above
3. Create a new analysis folder in the main directory. This is equivalent to the ```Pseudomonas``` directory
4. Within your analysis folder create ```data/``` directory and ```input/```, ```metadata/``` subdirectories
5. Copy ```Pseudomonas_sample_lvl_sim.ipynb``` and ```Pseudomonas_experiment_lvl_sim.ipynb``` into your respective folder.  

Next we need to modify the code for your analysis:
1. Update the ```config_Pa_sample_limma.tsv``` and ```config_Pa_experiment_limma.tsv``` based on your analysis
2. Update the analysis notebooks to use your config file and input file
3. Add your gene expression data file to the ```data/input/``` directory.  Your data is expected to be stored as a tab-delimited dataset with samples as rows and genes as columns.
4. Add your metadata file to ```data/metadata/``` directory.  Your metadata is expected to be stored as a tab-delimited with sample ids matching the gene expression dataset as one column and experiment ids as another. 
5. The module ```create_experiment_id_file``` in ```pipeline.py``` will need to be updated based on the format of your metadata file
6. Run notebooks

## Additional customization

Further customization can be accomplished by doing the following:

1. The ```apply_correction_io``` function in the ```functions/generate_data_parallel.py``` file can be modified to use a different correction method.
2. If there are additional pre-processing specific to your data, these can be added as modules in the ```pipeline.py``` file and called in the analysis notebook

## Configuration file

| Name | Description |
| :--- | :---------- |
| local_dir| str: Parent directory on local machine to store intermediate results|
| dataset_name| str: Name for analysis directory. Either "Human" or "Pseudomonas"|
| simulation_type | str: "sample_lvl_sim" or "experiment_lvl_sim"|
| NN_architecture | str: Name of neural network architecture to use. Format 'NN_<intermediate layer>_<latent layer>'|
| learning_rate| float: Parent directory on local machine to store intermediate results|
| batch_size | str: Name for analysis directory. Either "Human" or "Pseudomonas"|
| epochs | int: Number of times to train over the entire input dataset|
| kappa | float: How fast to linearly ramp up KL loss|
| intermediate_dim| int: Size of the hidden layer|
| latent_dim | int: Size of the bottleneck layer|
| epsilond_std | float: Standard deviation of Normal distribution to sample latent space|
| num_simulated_samples | int: Simulate a compendia with these many samples|
| num_simulated_experiments| int: Simulate a compendia with these many experiments|
| lst_num_experiments | list:  List of different numbers of experiments to add to simulated data.  These are the number of sources of technical variation that are added to the simulated data|
| lst_num_partitions | list:  List of different numbers of partitions to add to simulated data.  These are the number of sources of technical variation that are added to the simulated data|
| use_pca | bool: True if want to represent expression data in top PCs before calculating SVCCA similarity|
| num_PCs | int: Number of top PCs to use to represent expression data|
| correction_method | str: Noise correction method to use. Either "limma" or "combat"|
| metadata_colname | str: Column header that contains sample id that maps expression data and metadata|
| iterations | int: Number of simulations to run|
| num_cores | int: Number of processing cores to use|

## Acknowledgements
We would like to thank YoSon Park, David Nicholson, Ben Heil and Ariel Hippen-Anderson for insightful discussions and code review