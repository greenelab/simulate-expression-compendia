
# coding: utf-8

# # Simulation experiment using noise-corrected data
# 
# Run entire simulation experiment multiple times to generate confidence interval.  The simulation experiment can be found in ```functions/pipeline.py```

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

sys.path.append("../../")
from functions import pipelines, utils

from numpy.random import seed
randomState = 123
seed(randomState)


# In[ ]:


# Read in config variables
config_file = os.path.abspath(os.path.join(os.getcwd(),"../../configs", "config_Pa_sample.tsv"))
params = utils.read_config(config_file)


# In[ ]:


# Load parameters
dataset_name = params["dataset_name"]
analysis_name = params["analysis_name"]
NN_architecture = params["NN_architecture"]
num_simulated_samples = params["num_simulated_samples"]
lst_num_experiments = params["lst_num_experiments"]
use_pca = params["use_pca"]
num_PCs = params["num_PCs"]
local_dir = params["local_dir"]

iterations = params["iterations"] 
num_cores = params["num_cores"]


# In[ ]:


# Additional parameters
file_prefix = "Experiment_corrected"
corrected = True


# In[3]:


# Input files
base_dir = os.path.abspath(
  os.path.join(
      os.getcwd(), "../.."))    # base dir on repo


normalized_data_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "input",
    "train_set_normalized.pcl")


# In[4]:


# Output files
similarity_corrected_file = os.path.join(
    base_dir,
    "results",
    "saved_variables",
    dataset_name +"_sample_lvl_sim_similarity_corrected.pickle")

ci_corrected_file = os.path.join(
    base_dir,
    "results",
    "saved_variables",
    dataset_name +"_sample_lvl_sim_ci_corrected.pickle")


# In[ ]:


# Run multiple simulations
results = Parallel(n_jobs=num_cores, verbose=100)(
    delayed(
        pipelines.sample_level_simulation_corrected)(i,
                                                     NN_architecture,
                                                     dataset_name,
                                                     analysis_name,
                                                     num_simulated_samples,
                                                     lst_num_experiments,
                                                     corrected,
                                                     use_pca,
                                                     num_PCs,
                                                     file_prefix,
                                                     normalized_data_file,
                                                     local_dir) for i in iterations)


# In[ ]:


# Concatenate output dataframes
all_svcca_scores = pd.DataFrame()

for i in iterations:
    all_svcca_scores = pd.concat([all_svcca_scores, results[i][1]], axis=1)

all_svcca_scores


# In[ ]:


# Get median for each row (number of experiments)
mean_scores = all_svcca_scores.mean(axis=1).to_frame()
mean_scores.columns = ['score']
mean_scores


# In[ ]:


# Get standard dev for each row (number of experiments)
import math
std_scores = (all_svcca_scores.std(axis=1)/math.sqrt(10)).to_frame()
std_scores.columns = ['score']
std_scores


# In[ ]:


# Get confidence interval for each row (number of experiments)
# z-score for 95% confidence interval
err = std_scores*1.96


# In[ ]:


# Get boundaries of confidence interval
ymax = mean_scores + err
ymin = mean_scores - err

ci = pd.concat([ymin, ymax], axis=1)
ci.columns = ['ymin', 'ymax']
ci


# In[ ]:


mean_scores


# In[ ]:


# Pickle dataframe of mean scores scores for first run, interval
mean_scores.to_pickle(similarity_corrected_file)
ci.to_pickle(ci_corrected_file)

