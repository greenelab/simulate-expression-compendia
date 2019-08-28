# Batch effects simulation

**Background:**
In general, datasets tend to be processed in batches where subsets of samples are processed together.  When the data is processed in batches, each batch shares the same technical details such as the same technician, array platform, time of day, lab and study.  These factors are referred to as batch effects and introduce technical variation into the dataset that is separate from the biological variation in the data.  For example, say we were interested in identifying differentially expressed genes between disease versus normal samples.  However, we have gene expression measurements that were generated from two different lab technicians which can introduce variation between samples that are not due to disease state but rather slight differences in data handling between the two technicians.  We want to be able to normalize out this technical variation in the data in order to detect the variation that is due to disease versus normal.  

**Question:**
Batch effects are expected to obscure the biological signal that unsupervised analyses extract.  

If we have a compendium comprised of a collection of gene expression experiments, would we want a small or large number of experiments?

In other words, if we have a collection of gene expression data with technical noise that is experiment specific and biological signal that is more consistent, how are unsupervised learning approaches affected?  

**Hypothesis:**
Without any batches, the underlying signal is clear. With a few large batches, the batch effects are captured reducing the capacity of the model to extract biological features. With many batches, the underlying signal is again clearer.

**Impact:**
Assist in designing experiments.

**Approach:**
1. Use VAE to simulate realistic gene expression data
2. Add different amounts and types of batch effects
3. Compare the similarity between the input data and the batched data

# Experiments

[Experiment_0](https://github.com/ajlee21/Batch_effects_simulation/tree/master/scripts/experiment_0)  

**Goals:** 
To setup the pipeline that follows the approach enumerated above.

**Conclusions:**
In order to validate our approach, we examined different 2-layer VAE architectures and asked: Does our simulated data represent realistic gene expression data?  By visual inspection between our original input data (Pa compendium) vs simulated data, the overall structure is maintained as seen in our [2_simulate_data.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_0/2_simulate_data.ipynb)

[Experiment_1](https://github.com/ajlee21/Batch_effects_simulation/tree/master/scripts/experiment_1)

**Goal:**
To validate 1) SVCCA and 2) implementation of batch effects.  Specifically we are checking:

1.  Is our similarity metric, [SVCCA](https://arxiv.org/pdf/1706.05806.pdf), working as we expect? See [test_svcca_and_transformations.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_1/test_svcca_and_transformations.ipynb) 

2.  Does input dimensions affect the similarity calculation? See [test_svcca_and_dimensionality.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_1/test_svcca_and_dimensionality.ipynb).  We also used the same set of scripts from Experiment 0 but modified the simulation script, [2_simulate_data_truncated.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_1/2_simulate_data_truncated.ipynb), in order to allow the user to subsample the number of genes from the simulated dataset and examine the SVCCA performance. 

3.  We also tested different definitions of batch effects - mainly varying the strength of the batch effect.  In general, this translates to varying the ```stretch_factor``` variable to be large or small in [3_add_batch_effects.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_1/3_add_batch_effects.ipynb)

**Conclusions:**
1. Fewer input dimensions yields svcca score closer to 1 comparing dataset vs itself and a lower svcca score comparing dataset vs permuted dataset, as expected
2. Similarity score approaches the negative control (svcca score comparing dataset vs permuted dataset), which would indicate that as we increase the number of batch effects added, we are getting closer to noise.  It doesn’t appear that our similarity score is detecting our biological signal.  


[Experiment_2](https://github.com/ajlee21/Batch_effects_simulation/tree/master/scripts/experiment_2)

**Goal:**
To explore alternative similarity metrics including:
1. Visualizing data on PCA dimensions.  See [4_similarity_analysis_viz.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_2/4_similarity_analysis_viz.ipynb) for details.
2. Calculating CCA.  See [4_similarity_analysis_cca.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_2/4_similarity_analysis_cca.ipynb) for details.
3. Procrustes analysis.  See [4_similarity_analysis_procrustes.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_2/4_similarity_analysis_procrustes.ipynb) for details.
4. Calculating Hausdorff distance  See [4_similarity_analysis_hausdorff.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_2/4_similarity_analysis_hausdorff.ipynb) for details.

This experiment also modified the definition of batch effects by,
1. Shifting *all* genes using a gaussian distribution centered around 0 in order to shift gene expression in different directions as opposed to just one.  See [3_add_batch_effects.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_2/3_add_batch_effects.ipynb).
2. Verifying (via print statments) that the the subset of genes selected to be changed a) vary **between** batches and b) that the first batch is shifted from the original simulated.