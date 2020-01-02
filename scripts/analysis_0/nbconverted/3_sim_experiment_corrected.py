
# coding: utf-8

# # Simulation experiment 
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

import warnings
warnings.filterwarnings(action='ignore')

sys.path.append("../")
from functions import pipelines

from numpy.random import seed
randomState = 123
seed(randomState)


# In[ ]:


# Parameters
NN_architecture = 'NN_2500_30'
analysis_name = 'analysis_0'
file_prefix = 'Experiment_corrected'
num_simulated_samples = 6000
lst_num_experiments = [1, 2, 5, 10, 20,
                     50, 100, 500, 1000, 2000, 3000, 6000]
corrected = True
use_pca = True
num_PCs = 10

iterations = range(5) 
num_cores = 5


# In[ ]:


# Input files
base_dir = os.path.abspath(
  os.path.join(
      os.getcwd(), "../.."))    # base dir on repo


normalized_data_file = os.path.join(
  base_dir,
  "data",
  "input",
  "train_set_normalized.pcl")


# In[2]:


# Output files
local_dir = "/home/alexandra/Documents/"

similarity_corrected_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "output",
    "analysis_0_similarity_corrected.pickle")

ci_corrected_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "output",
    "analysis_0_ci_corrected.pickle")


# In[3]:


# Run multiple simulations - corrected
results = Parallel(n_jobs=num_cores, verbose=100)(
    delayed(
        pipelines.simple_simulation_experiment_corrected)(i,
                                                          NN_architecture,
                                                          analysis_name,
                                                          num_simulated_samples,
                                                          lst_num_experiments,
                                                          corrected,
                                                          use_pca,
                                                          num_PCs,
                                                          "Experiment",
                                                          normalized_data_file) for i in iterations)


# In[4]:


# Concatenate output dataframes
all_svcca_scores = pd.DataFrame()

for i in iterations:
    all_svcca_scores = pd.concat([all_svcca_scores, results[i][1]], axis=1)

all_svcca_scores


# In[5]:


# Get median for each row (number of experiments)
mean_scores = all_svcca_scores.mean(axis=1).to_frame()
mean_scores.columns = ['score']
mean_scores


# In[6]:


# Get standard dev for each row (number of experiments)
import math
std_scores = (all_svcca_scores.std(axis=1)/math.sqrt(10)).to_frame()
std_scores.columns = ['score']
std_scores


# In[7]:


# Get confidence interval for each row (number of experiments)
# z-score for 95% confidence interval
err = std_scores*1.96


# In[8]:


# Get boundaries of confidence interval
ymax = mean_scores + err
ymin = mean_scores - err

ci = pd.concat([ymin, ymax], axis=1)
ci.columns = ['ymin', 'ymax']
ci


# In[9]:


mean_scores


# In[10]:


# Pickle dataframe of mean scores scores for first run, interval
mean_scores.to_pickle(similarity_corrected_file)
ci.to_pickle(ci_corrected_file)

