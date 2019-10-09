# The impact of undesired technical variability on large-scale data compendia

**Motivation:**  
In the last two decades, scientists working in different labs have assayed gene expression from millions of samples. These experiments can be combined into a compendium in order to gain a systematic understanding of biological processes. However, combining different experiments introduces technical variance, which could distort biological signals in the data leading to misinterpretation. 

**Challenge:**
As the scale of these compendia increase, it becomes crucial to determine how integrating multiple experiments will disrupt our ability to detect biological patterns.

**Objective:**
We sought to determine the extent to which underlying biological signal can be extracted in the presence of technical artifacts via simulation. 

**Approach:**
We used a series of simulation experiments in order to answer the question: given a compendium of gene expression experiments, how will increasing the number of experiments change our ability to detect the original biological signal?

1. Train a multi-layer Variational Autoencoder (VAE) using [compendium](https://msystems.asm.org/content/1/1/e00025-15) of *P. aeruginosa* gene expression experiments from different labs measuring different biological processes
2. Used trained VAE to simulate a compendium with varying numbers of experiments
3. Compared gene expression patterns from a compendium containing a single simulated experiment and the pattern from a compendium with multiple experiments using a [singular vector canonical correlation (SVCCA)](https://arxiv.org/abs/1706.05806) analysis
4. Correct for technical variation added using existing methods like [removeBatchEffect](https://rdrr.io/bioc/limma/man/removeBatchEffect.html) and re-calculate similarity in reference to single simulated experiment

## Simulation experiments

### Simulation experiment #1.
Scripts can be found in [scripts/analysis_0](https://github.com/ajlee21/Batch_effects_simulation/tree/master/scripts/analysis_0)  

**Goal:** 
To assess the impact of technical variation from increasing numbers of experiments in the simplest case.

**Assumptions:**
1. There is linear technical noise between different experiments where the noise is normally distributed
2. All experiments have the same number of samples
3. The total number of samples in the compendium is fixed and each experiment has the same number of samples (i.e. the number of samples in each experiment is equal to the total number of samples in the compendium divided by the number of experiments).  In order to test for the effect of the number of experiments we held sample size fixed.
4. In order to generate new data we sample from latent space that represents an entire Pseudomonas compendium, and therefore represents a generic Pseudomonas expression signal

**Conclusions:**
We generated gene expression data that represents a collection of different biological signals.  We then added variation to samples in order to represent different numbers of experiments.  We found that having 2-100 experiments confound our ability to extract our original biological signal.  However, interestingly, as the number of experiments grows to hundreds it becomes easier to discover the underlying biological patterns.  Our ability to extract biological signal is defined by our ability to retain the structure of the biological data -- in other words, is our representation of the simulated data with a single experiment similar to the representation with multiple experiments?  

![Similarity](https://raw.githubusercontent.com/greenelab/Batch_effects_simulation/master/similarity_trend.png)

After applying linear correction method, removeBatchEffect, we find that for 2-100 simulated experiments we can correct for the variation added.  As we increase the number of experiments, each experiment has fewer samples where the technical variation signal is not as pronounced and so we start to remove the original signal (i.e. with 6000 experiments, each sample is an experiment and so the correction removes the entire signal that is present)
 
![Similarity_correction](https://raw.githubusercontent.com/greenelab/Batch_effects_simulation/master/similarity_after_correction.png)

### Simulation experiment #2.
Scripts can be found in [scripts/analysis_1](https://github.com/ajlee21/Batch_effects_simulation/tree/master/scripts/analysis_1)  

**Goal:** 
To assess the impact of technical variation from increasing numbers of experiments in the more complex case, in order to generate more realistic looking gene expression data.

**Assumptions:**
1. Simulated experiments will have the same structure as the existing Pseudomonas compendium experiments (i.e. the relationship between samples from the same experiment will be preserved in our simulation)
2. Simulated experiments will represent a different set of pertrubations (i.e. the level of gene expression activity will be different between the simulated vs the original dataset)