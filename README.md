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

There are 2 analyses using Pseudomonas dataset in the `Pseudomonas_analysis` directory and 2 analyses using the recount2 dataset in the `Human_analysis` directory:

| Name | Description |
| :--- | :---------- |
| [Pa_sample_lvl_sim](Pseudomonas_analysis/Pa_sample_lvl_sim/) | Pseudomonas sample-level gene expression simulation|
| [Pa_experiment_lvl_sim](Pseudomonas_analysis/Pa_experiment_lvl_sim/) | Pseudomonas experiment-level gene expression simulation|
| [Human_sample_lvl_sim](Human_analysis/Human_sample_lvl_sim/) | Human sample-level gene expression simulation|
| [Human_experiment_lvl_sim](Human_analysis/Human_experiment_lvl_sim/) | Human experiment-level gene expression simulation|


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
3. Create a new analysis folder in the main directory (i.e. "NAME_analysis")
4. Within your analysis folder add the same folders as found in ```Pseudomonas_analysis``` (i.e. experiment_lvl_sim/, sample_lvl_sim/, data/, logs/, models/)
5. Copy contents of ```Pseudomonas_analysis/Pa_sample_lvl_sim/``` into your respective folder.  
6. Copy contents of ```Pseudomonas_analysis/Pa_experiment_lvl_sim/``` into your respective folder.  

Next we need to modify the code for your analysis:
1. Update the ```config.tsv``` file your parameter settings
2. Add your gene expression data file to the ```data/input/``` directory.  Your data is expected to be stored as a tab-delimited dataset with samples as rows and genes as columns.
3. Add your metadata file to ```data/metadata/``` directory.  Your metadata is expected to be stored as a tab-delimited with sample ids matching the gene expression dataset as one column and experiment ids as another.
4. The ```get_sample_ids``` function in the ```functions/generate_data_parallel.py``` file needs to be modified to parse your metadata file provided.
5. Run all scripts in order

## How to run this simulation using your own data AND different noise correction method

The same steps above need to be performed with an additional modification for the correction method:

First we need to set up your local repository: 
1. Clone the ```simulate-expression-compendia``` repository
2. Set up conda environment using the command above
3. Create a new analysis folder in the main directory (i.e. "NAME_analysis")
4. Within your analysis folder add the same folder as found in ```Pseudomonas_analysis``` (i.e. experiment_lvl_sim/, sample_lvl_sim/, data/, logs/, models/)
5. Copy contents of ```Pseudomonas_analysis/Pa_sample_lvl_sim/``` into your respective folder.  
6. Copy contents of ```Pseudomonas_analysis/Pa_experiment_lvl_sim/``` into your respective folder.  

Next we need to modify the code for your analysis:
1. Update the ```config.tsv``` file your parameter settings
2. Add your gene expression data file to the ```data/input/``` directory.  Your data is expected to be stored as a tab-delimited dataset with samples as rows and genes as columns.
3. Add your metadata file to ```data/metadata/``` directory.  Your metadata is expected to be stored as a tab-delimited with sample ids matching the gene expression dataset as one column and experiment ids as another.
4. The ```get_sample_ids``` function in the ```functions/generate_data_parallel.py``` file needs to be modified to parse your metadata file provided.
5. The ```apply_correction_io``` function in the ```functions/generate_data_parallel.py``` file needs to be modified to use the different correction method.
6. Run all scripts in order


## Acknowledgements
We would like to thank YoSon Park, David Nicholson and Ben Heil for insightful discussions and code review