
# coding: utf-8

# # Generate data and calculate similarity
# 
# The goal of this notebook is to determine how much of the structure in the original dataset (single experiment) is retained after adding some number of experiments.
# 
# The approach is to,
# 1. Generates simulated data by sampling from a trained VAE model.  Simulate ```num_simulated_samples```
# 2. Add number of experiments in ```lst_num_experiments```
# 3. Calculate the similarity between the dataset with a single experiment and the dataset with some number of experiments added.  

# In[30]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import glob
import pandas as pd
import numpy as np
from plotnine import ggplot, ggtitle, xlab, ylab, geom_point, geom_line, aes, ggsave, scale_x_log10
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../")
from functions import generate_data
from functions import similarity_metric

from numpy.random import seed
randomState = 123
seed(randomState)


# In[22]:


# User parameters
NN_architecture = 'NN_2500_30'
analysis_name = 'analysis_0'
num_simulated_samples = 6000
lst_num_experiments = [1,2,5,10,20,50,100,500,1000,2000,3000,6000]
use_pca = True
num_PCs = 10


# In[3]:


# Input files
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))    # base dir on repo
local_dir = "/home/alexandra/Documents/"                         # base dir on local machine for data storage
                                                                 # Save doesn't recognize ~

normalized_data_file = os.path.join(
    base_dir,
    "data",
    "input",
    "train_set_normalized.pcl")


# In[4]:


# Output file
svcca_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "output",
    "svcca.pdf")


# ### Generate simulated data

# In[5]:


# Generate simulated data
generate_data.simulate_data(normalized_data_file,
                            NN_architecture,
                            analysis_name,
                            num_simulated_samples
                           )


# In[6]:


# Simulated data file 
simulated_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "simulated_data.txt.xz")


# ### Generate permuted version of simulated data (negative control)

# In[7]:


# Permute simulated data to be used as a negative control
generate_data.permute_data(simulated_data_file,
                          local_dir,
                          analysis_name)


# In[8]:


# Permuted simulated data file 
permuted_simulated_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "simulated",
    analysis_name,
    "permuted_simulated_data.txt.xz")


# ### Add number of experiments to simulated data
# 
# Note: The original Pseudomonas compendium contains 107 experiments (see data/metadata/sample_annotations.tsv).  For this simulation experiment, we are defining a simulated experiment to be a randomly subset of the ```num_simulated_samples``` simulated samples.  
# 
# For example, *experiment 1* = all simulated samples in 1 partition, experiment 2 = all samples partitioned into 2 experiments where each experiment has a different amount of random noise added to it.  This noise represents the technical variation for that experiment.

# In[9]:


# Add batch effects
generate_data.add_experiments(simulated_data_file,
                               lst_num_experiments,
                               local_dir,
                               analysis_name)


# ### Calculate similarity

# In[10]:


# Calculate similarity
batch_scores, permuted_score = similarity_metric.sim_svcca(simulated_data_file,
                                                           permuted_simulated_data_file,
                                                           'Experiment',
                                                           lst_num_experiments,
                                                           use_pca,
                                                           num_PCs,
                                                           local_dir,
                                                           analysis_name)


# In[26]:


# Convert similarity scores to pandas dataframe
similarity_score_df = pd.DataFrame(data={'score': batch_scores},
                                     index=lst_num_experiments,
                                    columns=['score'])
similarity_score_df.index.name = 'number of experiments'
similarity_score_df


# In[29]:


print("Similarity between input vs permuted data is {}".format(permuted_score))


# In[32]:


# Plot
threshold = pd.DataFrame(
    pd.np.tile(
        permuted_score,
        (len(lst_num_experiments), 1)),
    index=lst_num_experiments,
    columns=['score'])

g = ggplot(similarity_score_df, aes(x=lst_num_experiments, y='score'))     + geom_line()     + geom_line(aes(x=lst_num_experiments, y='score'), threshold, linetype='dashed')     + xlab('Number of Experiments')     + ylab('Similarity score (SVCCA)')     + ggtitle('Similarity across varying numbers of experiments')

print(g)
ggsave(plot=g, filename=svcca_file, dpi=300)

