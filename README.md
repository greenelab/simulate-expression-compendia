# The impact of undesired technical variability on large-scale data compendia

**Motivation:**  
Technical sources of variation in gene expression data commonly arise from differences in technicians, array platforms, sampling time or lab.  These variations distort biological signals in the data and can potentially cause misinterpretations.  In the last two decades, scientists working in different labs on different experiments have assayed millions of samples.  These experiments are being combined into compendia in order to gain a more systematic understanding of some biological processes.  

**Challenge:**
As the scale of these compendia increase, it becomes crucial to determine how integrating multiple experiments will disrupt our ability to detect biological patterns.

**Objective:**
We sought to determine the extent to which underlying biological signal can be extracted in the presence of technical artifacts via simulation. 

**Approach:**
1. Train a multi-layer Variational Autoencoder (VAE) using [compendium](https://msystems.asm.org/content/1/1/e00025-15) of *P. aeruginosa* gene expression experiments from different labs measuring different biological processes
2. Simulated gene expression from a compendium using the trained VAE
2. Added variation to subsets of simulated samples in order to represent different numbers of experiments 
3. Compared gene expression patterns from a compendium containing a single simulated experiment and the pattern from a compendium with multiple experiments using a canonical correlation analysis

# Experiments

[Experiment_0](https://github.com/ajlee21/Batch_effects_simulation/tree/master/scripts/experiment_0)  

**Goals:** 
...