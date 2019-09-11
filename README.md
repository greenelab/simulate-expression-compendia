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
3. Added variation to subsets of simulated samples in order to represent different numbers of experiments 
4. Compared gene expression patterns from a compendium containing a single simulated experiment and the pattern from a compendium with multiple experiments using a canonical correlation analysis
5. Correct for technical variation added using existing methods like [removeBatchEffect](https://rdrr.io/bioc/limma/man/removeBatchEffect.html) and re-calculate similarity in reference to single simulated experiment

# Analysis

[Analysis_0](https://github.com/ajlee21/Batch_effects_simulation/tree/master/scripts/analysis_0)  

**Goals:** 
To demonstrate proof-of-concept that:
1. Simulating a compendium containing few experiments, it is difficult to detect the simulated signals of interest.  However, as the number of experiments increased, the simulated signals became more clear.
2. We can correct for this added technical variation from having multiple experiments.

**Conclusions:**
We generated gene expression data that represents a collection of different biological signals.  We then added variation to samples in order to represent different numbers of experiments.  We found that having 2-20 experiments confound our ability to extract our original biological signal.  However, interestingly, as the number of experiments grows to hundreds it becomes easier to discover the underlying biological patterns.  Our ability to extract biological signal is defined by our ability to retain the structure of the biological data -- in other words, is our representation of the simulated data with a single experiment similar to the representation with multiple experiments?  

![Test Image 1](https://github.com/ajlee21/Batch_effects_simulation/tree/master/similarity_trend.png)