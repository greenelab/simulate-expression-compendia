#!/usr/bin/env python
# coding: utf-8

# # Simulate pseudo experiments using random sampling
# 
# This notebook generates new pseudo-experiments by the randomly sampling from the [sample level simulated](../Pseudomonas/Pseudomonas_sample_lvl_sim.ipynb) compendium.  The expression patterns in these new experiments are used as a negative control against the patterns in the experiments generated in [generate_E_GEOD_51409_template_experiment.ipynb](generate_E_GEOD_51409_template_experiment.ipynb) in the [differential expression analysis](DE_analysis_run.R) and [pathway enrichment analysis](find_enrichment_run.R)

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import ast
import pandas as pd
import numpy as np
import seaborn as sns
import random
import glob
from sklearn import preprocessing

sys.path.append("../")
from ponyo import utils

import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


# Read in config variables
config_file = os.path.abspath(os.path.join(os.getcwd(),"../configs", "config_Pa_sample_limma.tsv"))
params = utils.read_config(config_file)


# In[3]:


# Load parameters
num_runs = 100
dataset_name = params["dataset_name"]
num_simulated_samples = params["num_simulated_samples"]
NN_architecture = params["NN_architecture"]
local_dir = params["local_dir"]


# In[4]:


# Input files
base_dir = os.path.abspath(
  os.path.join(
      os.getcwd(), "../"))    # base dir on repo

original_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "Pa_compendium_02.22.2014.pcl")

simulated_data_file = os.path.join(
    local_dir,
    "experiment_simulated",
    "Pseudomonas_sample_lvl_sim",
    "Experiment_1_0.txt.xz")

mapping_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "metadata",
    "sample_annotations.tsv")


# ## Process data
# 
# Notice: Originally the expression data was 0-1 normalized for use in training the VAE, however when we performed differential expression analyses we found that the normalized data had reduced variance that resulted in an inconsistency between the number of DEGs found compared to the publication. Therefore, we are re-scaling our normalized data to be in the original range of data.

# In[5]:


# Read data
original_data = pd.read_table(
    original_data_file,
    header=0,
    sep='\t',
    index_col=0).T

simulated_data = pd.read_table(
    simulated_data_file,
    header=0,
    sep='\t',
    index_col=0)

print(original_data.shape)
print(simulated_data.shape)


# In[6]:


original_data.head(5)


# In[7]:


simulated_data.head(5)


# In[8]:


# 0-1 normalize per gene
scaler = preprocessing.MinMaxScaler()

original_data_scaled = scaler.fit_transform(original_data)
normalized_data = pd.DataFrame(original_data_scaled,
                                columns=original_data.columns,
                                index=original_data.index)

normalized_data.head(5)


# In[9]:


# Re-scale simulated data back into the same range as the original data
simulated_data_scaled = scaler.inverse_transform(simulated_data)

simulated_data_scaled_df = pd.DataFrame(simulated_data_scaled,
                                columns=simulated_data.columns,
                                index=simulated_data.index)

simulated_data_scaled_df.head(5)


# In[10]:


# Read in metadata
metadata = pd.read_table(
    mapping_file, 
    header=0, 
    sep='\t', 
    index_col=0)

metadata.head()


# In[11]:


map_experiment_sample = metadata[['sample_name', 'ml_data_source']]
map_experiment_sample.head()


# # Use experiment E-GEOD-51409 as a template
# 
# This experiment measures the transcriptome of *P. aeruginosa* under two different growth conditions: 28 degrees and 37 degress. This experiment contains 6 total samples with 3 samples per condition.
# 
# As a control, we will simulate a pseudo-experiment by ranomly sampling from the compendia, which does **not** preserve the experiment structure (see module ```simulate_data``` in ```functions/generate_data_parallel.py```). We will sample 6 random samples and group them into 2 groups with 3 samples per group (i.e. following the same design as the template experiment). However, this pseudo-experiment is a set of random samples that ignores experiment structure and so we don't anticipate there to find much biological signficance in this pseudo-experiment.

# In[12]:


# Get experiment id
experiment_id = 'E-GEOD-51409'


# In[13]:


# Get original samples associated with experiment_id
selected_mapping = map_experiment_sample.loc[experiment_id]
original_selected_sample_ids = list(selected_mapping['ml_data_source'].values)


# In[14]:


# Create example random simulated
# Randomly select samples from simulated data
num_samples = len(original_selected_sample_ids)
selected_control_data = simulated_data_scaled_df.sample(n=num_samples)

# Map sample ids from original data to simulated data
selected_control_data.index = original_selected_sample_ids
selected_control_data.columns = normalized_data.columns

# Save selected samples
# This will be used as input into R script to identify differentially expressed genes
selected_control_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "selected_control_data_"+experiment_id+"_example.txt")

selected_control_data.to_csv(
    selected_control_data_file, float_format='%.3f', sep='\t')


# In[15]:


# Create multiple control datasets
for i in range(num_runs):
    # Randomly select samples from simulated data
    num_samples = len(original_selected_sample_ids)
    selected_control_data = simulated_data_scaled_df.sample(n=num_samples)
    
    # Map sample ids from original data to simulated data
    selected_control_data.index = original_selected_sample_ids
    selected_control_data.columns = normalized_data.columns
    
    # Save selected samples
    # This will be used as input into R script to identify differentially expressed genes
    selected_control_data_file = os.path.join(
        local_dir,
        "pseudo_experiment",
        "selected_control_data_"+experiment_id+"_"+str(i)+".txt")
    
    selected_control_data.to_csv(
        selected_control_data_file, float_format='%.3f', sep='\t')

