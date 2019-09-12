
# coding: utf-8

# # Generate data and calculate similarity
# 
# The goal of this notebook is to determine how much of the structure in the original dataset (single experiment) is retained after adding some number of experiments.
# 
# The approach is to,
# 1. Generates simulated data by sampling from a trained VAE model.  Simulate ```num_simulated_samples```
# 2. Add number of experiments in ```lst_num_experiments```
# 3. Calculate the similarity between the dataset with a single experiment and the dataset with some number of experiments added.  

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import glob
import pandas as pd
from plotnine import ggplot, ggtitle, xlab, ylab, geom_point, geom_line, aes
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../")
from functions import generate_data
from functions import similarity_metric

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# User parameters
NN_architecture = 'NN_2500_30'
analysis_name = 'analysis_0'
num_simulated_samples = 6000
lst_num_experiments = [1,2,5,10,20,50,100,500,1000,2000,3000,6000]
use_pca = True
num_PCs = 10


# In[3]:


# Input files
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../.."))

normalized_data_file = os.path.join(
    base_dir,
    "data",
    "input",
    "train_set_normalized.pcl")


# ### Generate simulated data

# In[4]:


# Generate simulated data
generate_data.simulate_data(normalized_data_file,
                            NN_architecture,
                            analysis_name,
                            num_simulated_samples
                           )


# In[5]:


# Simulated data file 
simulated_data_file = os.path.join(
    base_dir,
    "data",
    "simulated",
    analysis_name,
    "simulated_data.txt.xz")


# ### Generate permuted version of simulated data (negative control)

# In[6]:


# Permute simulated data to be used as a negative control
generate_data.permute_data(simulated_data_file,
                          base_dir,
                          analysis_name)


# In[7]:


# Permuted simulated data file 
permuted_simulated_data_file = os.path.join(
    base_dir,
    "data",
    "simulated",
    analysis_name,
    "permuted_simulated_data.txt.xz")


# ### Add number of experiments to simulated data

# In[8]:


# Add batch effects
generate_data.add_experiments(simulated_data_file,
                               lst_num_experiments,
                               base_dir,
                               analysis_name)


# ### Calculate similarity

# In[9]:


# Calculate similarity
batch_scores, permuted_score = similarity_metric.sim_svcca(simulated_data_file,
                                                           permuted_simulated_data_file,
                                                           lst_num_experiments,
                                                           use_pca,
                                                           num_PCs,
                                                           base_dir,
                                                           analysis_name)


# In[10]:


# Convert similarity scores to pandas dataframe
similarity_score_df = pd.DataFrame(data={'score': batch_scores},
                                     index=lst_num_experiments,
                                    columns=['score'])
similarity_score_df.index.name = 'number of batches'
similarity_score_df


# In[11]:


permuted_score


# In[12]:


# Plot
threshold = pd.DataFrame(
    pd.np.tile(
        permuted_score,
        (len(lst_num_experiments), 1)),
    index=lst_num_experiments,
    columns=['score'])

ggplot(similarity_score_df, aes(x=lst_num_experiments, y='score'))     + geom_line()     + geom_line(aes(x=lst_num_experiments, y='score'), threshold, linetype='dashed')     + xlab('Number of Experiments')     + ylab('SVCCA')     + ggtitle('Similarity across increasing batch effects')

