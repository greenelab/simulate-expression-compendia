# Simulation experiment using *Pseudomonas aeruginosa* (*P. aeruginosa*) dataset 

## Sample-level simulation experiment 
Scripts can be found in [Pa_sample_lvl_sim](https://github.com/ajlee21/Batch_effects_simulation/tree/master/Pseudomonas_analysis/Pa_sample_lvl_sim)  

**Goal:** 
To assess the impact of technical variation from increasing numbers of experiments in the simplest case.

**Assumptions:**
1. There is linear technical noise between different experiments where the noise is normally distributed
2. All experiments have the same number of samples
3. The total number of samples in the compendium is fixed and each experiment has the same number of samples (i.e. the number of samples in each experiment is equal to the total number of samples in the compendium divided by the number of experiments).  In order to test for the effect of the number of experiments we held sample size fixed.
4. In order to generate new data we sample from latent space that represents an entire Pseudomonas compendium, and therefore represents a generic Pseudomonas expression signal

**Approach:**
Given a compendium of gene expression experiments, how will increasing the number of experiments change our ability to detect the original underlying biological signal?

1. Train a multi-layer Variational Autoencoder (VAE) using [compendium](https://msystems.asm.org/content/1/1/e00025-15) of *P. aeruginosa* gene expression experiments from different labs measuring different biological processes
2. Used trained VAE to simulate a compendium by randomly sampling from the low dimensional latent space representation of the *P. aeruginosa* dataset.  
3. Add varying numbers of technical variation to the simulated compendium to generate multiple noisy compendia.
4. Compare gene expression patterns from the simulated compendium vs the pattern from a compendia with some number of technical variation added using a [singular vector canonical correlation (SVCCA)](https://arxiv.org/abs/1706.05806) analysis
5. Correct for the added technical variation using existing method [removeBatchEffect](https://rdrr.io/bioc/limma/man/removeBatchEffect.html) and calculate the SVCCA similarity of the corrected noisy compendia vs simulated compendium

**Conclusions:**
Having some number of technical variation confounded our ability to extract our original biological signal and we need to apply correction.  However, interestingly, as the number of technical variation increased, it became easier to discover the underlying biological patterns and applying correction removes some of the biology of interest.  Our ability to extract biological signal is defined by our ability to retain the structure of the biological data -- in other words, is our representation of the simulated data with a single experiment similar to the representation with multiple experiments?  

![Similarity](https://raw.githubusercontent.com/greenelab/Batch_effects_simulation/master/results/Pa_sample_lvl_sim_svcca.svg)


## Experiment-level simulation experiment 
Scripts can be found in [scripts/analysis_1](https://github.com/ajlee21/Batch_effects_simulation/tree/master/Pseudomonas_analysis/Pa_experiment_lvl_sim)  

**Goal:** 
To assess the impact of technical variation from increasing numbers of experiments in the more complex case, in order to generate more realistic looking gene expression data.

**Assumptions:**
1. Simulated experiments will have the same structure as the existing Pseudomonas compendium experiments (i.e. the relationship between samples from the same experiment will be preserved in our simulation)
2. Simulated experiments will represent a different set of pertrubations (i.e. the level of gene expression activity will be different between the simulated vs the original dataset)

**Approach:**
Given a compendium of gene expression experiments, how will increasing the number of experiments change our ability to detect the original underlying biological signal?

1. Train a multi-layer Variational Autoencoder (VAE) using [compendium](https://msystems.asm.org/content/1/1/e00025-15) of *P. aeruginosa* gene expression experiments from different labs measuring different biological processes
2. Used trained VAE to simulate a compendium by randomly sampling **experiments** from the low dimensional latent space representation of the *P. aeruginosa* dataset and shifting the samples from the experiments within the space.  This process creates a simulated compendium of different experiments of similar type to the original but with different perturbations.
3. Add varying numbers of technical variation to the simulated compendium, adding technical variation per experiment, to generate multiple noisy compendia.
4. Compare gene expression patterns from the simulated compendium vs the pattern from a compendia with some number of technical variation added using a [singular vector canonical correlation (SVCCA)](https://arxiv.org/abs/1706.05806) analysis
5. Correct for the added technical variation using existing method [removeBatchEffect](https://rdrr.io/bioc/limma/man/removeBatchEffect.html) and calculate the SVCCA similarity of the corrected noisy compendia vs simulated compendium

**Conclusions:**
We observed the same trend as before.  

![Similarity](https://raw.githubusercontent.com/greenelab/Batch_effects_simulation/master/results/Pa_experiment_lvl_sim_svcca.svg)