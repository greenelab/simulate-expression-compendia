# Simulation experiment using recount2 dataset 

## Sample-level simulation experiment 
Scripts can be found in [Human_sample_lvl_sim](https://github.com/greenelab/simulate-expression-compendia/tree/master/Human_analysis/Human_sample_lvl_sim)  

**Goal:** 
To assess the impact of technical variation from increasing numbers of partitions in the simplest case.

**Assumptions:**
1. There is linear technical noise between different partitions where the noise is normally distributed
2. The total number of samples in the compendium is fixed (i.e. the number of samples in each partition is equal to the total number of samples in the compendium divided by the number of partitions).  In order to test for the effect of the number of experiments we held sample size fixed.
3. In order to generate new data we sample from latent space that represents an entire Pseudomonas compendium, and therefore represents a generic Pseudomonas expression signal

**Approach:**
Given a compendium of gene expression partitions, how will increasing the number of partitions change our ability to detect the original underlying biological signal?

1. Train a multi-layer Variational Autoencoder (VAE) using a [compendium](https://www.refine.bio) of human gene expression experiments from different labs measuring different biological processes
2. Use trained VAE to simulate a compendium by randomly sampling from the low dimensional latent space representation of the *P. aeruginosa* dataset.  
3. Randomly divide samples into partitions.  Here partitions represent a random group of samples that are from the same experiment or were generated from the same lab.
4. Add varying numbers of sources of technical variation to each partition to generate multiple noisy compendia.
4. Compare gene expression patterns from the simulated compendium vs the pattern from a compendia with some number of technical variation added using a [singular vector canonical correlation (SVCCA)](https://arxiv.org/abs/1706.05806) analysis
5. Correct for the added technical variation using existing method [removeBatchEffect](https://rdrr.io/bioc/limma/man/removeBatchEffect.html) and calculate the SVCCA similarity of the corrected noisy compendia vs simulated compendium

**Conclusions:**
Having some number of technical variation confounded our ability to extract our original biological signal and we need to apply correction.  However, as the number of technical variation increased, it became easier to discover the underlying biological patterns and applying correction removes some of the biology of interest.  Our ability to extract biological signal is defined by our ability to retain the structure of the biological data -- in other words, is our representation of the simulated data with a single partition similar to the representation with multiple partitions?  

![Similarity_Human_sample](https://github.com/greenelab/simulate-expression-compendia/blob/master/results/Human_sample_lvl_sim_svcca.png)


## Experiment-level simulation experiment 
Scripts can be found in [scripts/Human_experiment_lvl_sim](https://github.com/greenelab/simulate-expression-compendia/tree/master/Human_analysis/Human_experiment_lvl_sim)  

**Goal:** 
To assess the impact of technical variation from increasing numbers of partitions in the more complex case, in order to generate more realistic looking gene expression data.

**Assumptions:**
1. Simulated experiments will have the same structure as the existing Pseudomonas compendium experiments (i.e. the relationship between samples from the same experiment will be preserved in our simulation)
2. Simulated experiments will represent a different set of pertrubations (i.e. the level of gene expression activity will be different between the simulated vs the original dataset)

**Approach:**
Given a compendium of gene expression partitions, how will increasing the number of partitions change our ability to detect the original underlying biological signal?

1. Train a multi-layer Variational Autoencoder (VAE) using [compendium](https://www.refine.bio) of human gene expression experiments from different labs measuring different biological processes
2. Used trained VAE to simulate a compendium by randomly sampling **experiments** from the low dimensional latent space representation of the *P. aeruginosa* dataset and shifting the samples from the experiments within the space.  This process creates a simulated compendium of different experiments of similar type to the original but with different perturbations.
3. Randomly divide new experiments into partitions.  Here partitions represent groups of experiments that were generated from the same lab or have the same experimental design.
4. Add varying numbers of sources of technical variation to each partition to generate multiple noisy compendia.
5. Compare gene expression patterns from the simulated compendium vs the pattern from a compendia with some number of technical variation added using a [singular vector canonical correlation (SVCCA)](https://arxiv.org/abs/1706.05806) analysis
6. Correct for the added technical variation using existing method [removeBatchEffect](https://rdrr.io/bioc/limma/man/removeBatchEffect.html) and calculate the SVCCA similarity of the corrected noisy compendia vs simulated compendium

**Conclusions:**
We observed the same trend as before.  

![Similarity_Human_experiment](https://github.com/greenelab/simulate-expression-compendia/blob/master/results/Human_experiment_lvl_sim_svcca.png)
