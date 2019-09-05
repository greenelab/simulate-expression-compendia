# Archived scripts

**Approach:**
1. Use VAE to simulate realistic gene expression data
2. Add different numbers of batch effects (i.e. experiments)
3. Compare the similarity between the input data and input data with batch effects added

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
1. Shifting *all* genes using a vector of values sampled from a gaussian distribution centered around 0.  We want to shift gene expression in different directions as opposed to just one.  See [3_add_batch_effects.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_2/3_add_batch_effects.ipynb).
2. Embedding our high dimensional gene expression data into PCA space and using this compressed representation to calculate similarity.  See [4_similarity_analysis_pca_svcca.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_2/4_similarity_analysis_pca_svcca.ipynb).
3. Verifying (via print statments) that the the subset of genes selected to be changed a) vary **between** batches and b) that the first batch is shifted from the original simulated.