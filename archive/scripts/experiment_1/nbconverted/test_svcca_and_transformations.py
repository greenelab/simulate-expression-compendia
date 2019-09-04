#!/usr/bin/env python
# coding: utf-8

# # SVCCA and affine transformations
# We want to test that SVCCA is working as expected.  In other words, what is the SVCCA score when we compare two datasets that are 1) identical and 2) one is a contant scaled version of the other?

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


# ## Similarity between input vs itself
# We expect that the similarity of SVCCA comparing the input with itself should yeild an SVCCA score of 1.0

# In[5]:


get_ipython().run_cell_magic('time', '', '# SVCCA\nsvcca_results = cca_core.get_cca_similarity(simulated_data.T,\n                                      simulated_data.T,\n                                      verbose=False)\n\nprint(np.mean(svcca_results["cca_coef1"]))')


# ## Similarity between input vs scaled version of input¶
# We expect that the similarity of SVCCA comparing the input with scaled version of itself to yield a high SVCCA score since this transformation is an affine transformation which SVCCA is supposed to be invariant to.

# In[6]:


# Scale data by a constant
scaled_simulated_data = simulated_data.multiply(2)
scaled_simulated_data.head()


# In[7]:


get_ipython().run_cell_magic('time', '', '# SVCCA\nsvcca_results = cca_core.get_cca_similarity(simulated_data.T,\n                                      scaled_simulated_data.T,\n                                      verbose=False)\n\nprint(np.mean(svcca_results["cca_coef1"]))')


# ## Similarity between input vs permuted input¶
# We will use the similarity of between the input with permuted input as a negative control.  We would expect the SVCCA score to be fairly low for this comparison.

# In[8]:


get_ipython().run_cell_magic('time', '', '# Permute simulated data\nshuffled_simulated_arr = []\nnum_samples = simulated_data.shape[0]\n\nfor i in range(num_samples):\n    row = list(simulated_data.values[i])\n    shuffled_simulated_row = random.sample(row, len(row))\n    shuffled_simulated_arr.append(shuffled_simulated_row)')


# In[12]:


shuffled_simulated_data = pd.DataFrame(shuffled_simulated_arr, index=simulated_data.index, columns=simulated_data.columns)
shuffled_simulated_data.head()


# In[11]:


get_ipython().run_cell_magic('time', '', '# SVCCA\nsvcca_results = cca_core.get_cca_similarity(simulated_data.T,\n                                      shuffled_simulated_data.T,\n                                      verbose=False)\n\nprint(np.mean(svcca_results["cca_coef1"]))')


# ## Toy

# In[21]:


df = pd.DataFrame({'A': range(5),'B': np.ones(5), 'C': [20,40,60,80,100], 'D': np.ones(5)*5})
df.head()


# In[22]:


shuffled_arr = []


for i in range(len(df.values)):
    row = list(df.values[i])
    shuffled = random.sample(row, len(row))
    shuffled_arr.append(shuffled)
shuffled_df = pd.DataFrame(shuffled_arr, index=df.index, columns=df.columns)
shuffled_df.head()

