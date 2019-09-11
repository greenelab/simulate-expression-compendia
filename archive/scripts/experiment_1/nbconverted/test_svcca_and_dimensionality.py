#!/usr/bin/env python
# coding: utf-8

# # SVCCA and dimensionality
# We want to test the affect of the input data dimensions on SVCCA performance.  As we increase the number of dimensions how does SVCCA change?  

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import numpy as np
import random
import glob
import warnings
warnings.filterwarnings("ignore")

from ggplot import *
from functions import cca_core

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Parameters
analysis_name = 'experiment_0'
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))
num_dims = [10, 100, 1000, 2000, 3000, 4000, 5549]


# In[3]:


# Load arguments
simulated_data_file = os.path.join(
    base_dir,
    "data",
    "simulated",
    analysis_name,
    "simulated_data.txt")


# In[4]:


# Read in simulated gene expression data
simulated_data = pd.read_table(
    simulated_data_file,
    header=0, 
    index_col=0,
    sep='\t')

simulated_data.head()


# # Ordered sampling
# 
# ## Similarity between input vs itself
# We expect that the similarity of SVCCA comparing the input with itself should yeild an SVCCA score of 1.0

# In[5]:


get_ipython().run_cell_magic('time', '', '# SVCCA\nsvcca_scores_itself = []\nfor z in num_dims:\n    subset_simulated_data = simulated_data.iloc[:,0:z]\n    svcca_results = cca_core.get_cca_similarity(subset_simulated_data.T,\n                                          subset_simulated_data.T,\n                                          verbose=False)\n\n    svcca_scores_itself.append(np.mean(svcca_results["cca_coef1"]))\n    print("{} dimensions ... {}".format(z,np.mean(svcca_results["cca_coef1"])))\n    \n#print(svcca_scores_itself)')


# ## Similarity between input vs permuted input¶
# We will use the similarity of between the input with permuted input as a negative control.  We would expect the SVCCA score to be fairly low for this comparison.

# In[6]:


get_ipython().run_cell_magic('time', '', '# Permute simulated data\nshuffled_simulated_arr = []\nnum_samples = simulated_data.shape[0]\n\nfor i in range(num_samples):\n    row = list(simulated_data.values[i])\n    shuffled_simulated_row = random.sample(row, len(row))\n    shuffled_simulated_arr.append(shuffled_simulated_row)\n    \nshuffled_simulated_data = pd.DataFrame(shuffled_simulated_arr, index=simulated_data.index, columns=simulated_data.columns)\nshuffled_simulated_data.head()')


# In[7]:


get_ipython().run_cell_magic('time', '', '# SVCCA\nsvcca_scores_shuffled = []\nfor z in num_dims:\n    subset_simulated_data = simulated_data.iloc[:,0:z]\n    subset_shuffled_simulated_data = shuffled_simulated_data.iloc[:,0:z]\n    svcca_results = cca_core.get_cca_similarity(subset_simulated_data.T,\n                                          subset_shuffled_simulated_data.T,\n                                          verbose=False)\n\n    svcca_scores_shuffled.append(np.mean(svcca_results["cca_coef1"]))\n    print("{} dimensions ... {}".format(z,np.mean(svcca_results["cca_coef1"])))\n    \n#print(svcca_scores_shuffled)')


# # Random sampling
# 
# ## Similarity between input vs itself and input vs permuted input
# Perform the same analysis as above, this time taking random samples instead of ordered samples

# In[5]:


get_ipython().run_cell_magic('time', '', '\n# Store svcca scores\nsvcca_scores_itself = []\nsvcca_scores_shuffled = []\n\nfor z in num_dims:\n    \n    # Randomly select z dimensions \n    subset_simulated_data = simulated_data.sample(n=z, axis=1)\n    print(subset_simulated_data.head())\n    \n    # Permute subset of data\n    shuffled_simulated_arr = []\n    num_samples = subset_simulated_data.shape[0]\n\n    for i in range(num_samples):\n        row = list(subset_simulated_data.values[i])\n        shuffled_simulated_row = random.sample(row, len(row))\n        shuffled_simulated_arr.append(shuffled_simulated_row)\n\n    subset_shuffled_simulated_data = pd.DataFrame(shuffled_simulated_arr, \n                                           index=subset_simulated_data.index, \n                                           columns=subset_simulated_data.columns)\n    \n    print(subset_shuffled_simulated_data.head())\n    \n    # Calculate SVCCA for subset vs itself\n    svcca_results = cca_core.get_cca_similarity(subset_simulated_data.T,\n                                          subset_simulated_data.T,\n                                          verbose=False)\n\n    svcca_scores_itself.append(np.mean(svcca_results["cca_coef1"]))\n    print("{} dimensions ... SVCCA(itself) {}".format(z,np.mean(svcca_results["cca_coef1"])))\n    \n    # Calculate SVCCA for subset vs permuted subset\n    svcca_results = cca_core.get_cca_similarity(subset_simulated_data.T,\n                                          subset_shuffled_simulated_data.T,\n                                          verbose=False)\n\n    svcca_scores_shuffled.append(np.mean(svcca_results["cca_coef1"]))\n    print("{} dimensions ... SVCCA (permuted) {}".format(z,np.mean(svcca_results["cca_coef1"])))')


# **Observations**
# Looks like dimensionality affects SVCCA performance.  Comparing simulated data versus itself is most similar with fewer dimensions and decreases as we add dimensions.  Perhaps this indicates that the structure in the data is lost in such high dimensions.  
