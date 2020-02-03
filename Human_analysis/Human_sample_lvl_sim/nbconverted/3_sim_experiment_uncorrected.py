
# coding: utf-8

# # Simulation experiment using noisy data
# 
# Run entire simulation experiment multiple times to generate confidence interval

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from joblib import Parallel, delayed
import multiprocessing
import sys
import os
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../../")
from functions import pipelines

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Parameters
dataset_name = "Human_analysis"
analysis_name = 'Human_sample_lvl_sim'
NN_architecture = 'NN_2500_30'
file_prefix = "Experiment"
num_simulated_samples = 500
lst_num_experiments = [1, 2, 5, 10, 20,
                     50, 100, 250, 500]

corrected = False
use_pca = True
num_PCs = 10

iterations = range(5) 
num_cores = 5


# In[3]:


# Input file
base_dir = os.path.abspath(
      os.path.join(
          os.getcwd(), "../.."))

normalized_data_file = os.path.join(
    base_dir,
    dataset_name,    
    "data",
    "input",
    "recount2_gene_normalized_data.tsv.xz")


# In[4]:


# Output files
similarity_uncorrected_file = os.path.join(
    base_dir,
    "results",
    "saved_variables",
    "Human_sample_lvl_sim_similarity_uncorrected.pickle")

ci_uncorrected_file = os.path.join(
    base_dir,
    "results",
    "saved_variables",
    "Human_sample_lvl_sim_ci_uncorrected.pickle")

similarity_permuted_file = os.path.join(
    base_dir,
    "results",
    "saved_variables",
    "Human_sample_lvl_sim_permuted")


# In[5]:


# Run multiple simulations - corrected
results = Parallel(n_jobs=num_cores, verbose=100)(
    delayed(
        pipelines.sample_level_simulation_uncorrected)(i,
                                                       NN_architecture,
                                                       dataset_name,
                                                       analysis_name,
                                                       num_simulated_samples,
                                                       lst_num_experiments,
                                                       corrected,
                                                       use_pca,
                                                       num_PCs,
                                                       file_prefix,
                                                       normalized_data_file) for i in iterations)


# In[6]:


# permuted score
permuted_score = results[0][0]


# In[7]:


# Concatenate output dataframes
all_svcca_scores = pd.DataFrame()

for i in iterations:
    all_svcca_scores = pd.concat([all_svcca_scores, results[i][1]], axis=1)

all_svcca_scores


# In[8]:


# Get median for each row (number of experiments)
mean_scores = all_svcca_scores.mean(axis=1).to_frame()
mean_scores.columns = ['score']
mean_scores


# In[9]:


# Get standard dev for each row (number of experiments)
import math
std_scores = (all_svcca_scores.std(axis=1)/math.sqrt(10)).to_frame()
std_scores.columns = ['score']
std_scores


# In[10]:


# Get confidence interval for each row (number of experiments)
err = std_scores*1.96


# In[11]:


# Get boundaries of confidence interval
ymax = mean_scores + err
ymin = mean_scores - err

ci = pd.concat([ymin, ymax], axis=1)
ci.columns = ['ymin', 'ymax']
ci


# In[12]:


mean_scores


# In[13]:


# Pickle dataframe of mean scores scores for first run, interval
mean_scores.to_pickle(similarity_uncorrected_file)
ci.to_pickle(ci_uncorrected_file)
np.save(similarity_permuted_file, permuted_score)
