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

[Experiment_0](https://github.com/ajlee21/Batch_effects_simulation/tree/master/scripts/experiment_0)  is setting up the pipeline that follows the approach described above.  This experiment also examined different 2-layer VAE architectures and asked:

Does our simulated data represent realistic gene expression data?  By visual inspection between our original input data (Pa compendium) vs simulated data, the overall structure is maintained as seen in our [simulate_data.ipynb](https://github.com/ajlee21/Batch_effects_simulation/blob/master/scripts/experiment_0/2_simulate_data.ipynb)

[Experiment_1](https://github.com/ajlee21/Batch_effects_simulation/tree/master/scripts/experiment_1)is an experiment to validate SVCCA.  Specifically we are checking:

1.  Is our similarity metric, [SVCCA](https://arxiv.org/pdf/1706.05806.pdf), working as we expect?  
2.  Does input dimensions affect the similarity calculation?

