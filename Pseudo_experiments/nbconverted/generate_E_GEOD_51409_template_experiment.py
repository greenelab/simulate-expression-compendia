
# coding: utf-8

# # Simulate pseudo experiments using template experiment
# 
# This notebook generates new pseudo-experiments using the experiment-preserving approach in the [experiment level simulation](../Pseudomonas/Pseudomonas_experiment_lvl_sim.ipynb). In this simulation we are preserving the experiment type but not the actual experiment so the relationship between samples within an experiment are preserved but the genes that are expressed will be different (module [simulate_compendium](../functions/generate_data_parallel.py)).
# 
# The expression patterns in these new experiments are compared against the patterns in the experiments generated in [generate_random_sampled_experiment.ipynb](generate_random_sampled_experiment.ipynb) using [differential expression analysis](DE_analysis_run.R) and [pathway enrichment analysis](find_enrichment_run.R)

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
from functions import utils
import generate_labeled_data

import warnings
warnings.filterwarnings(action='ignore')

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Read in config variables
config_file = os.path.abspath(os.path.join(os.getcwd(),"../configs", "config_Pa_experiment_limma.tsv"))
params = utils.read_config(config_file)


# In[3]:


# Load parameters
num_runs = 100
dataset_name = params["dataset_name"]
num_simulated_experiments = params["num_simulated_experiments"]
NN_architecture = params["NN_architecture"]
local_dir = params["local_dir"]


# In[4]:


# Input files
base_dir = os.path.abspath(
  os.path.join(
      os.getcwd(), "../"))    # base dir on repo

# Load experiment id file
# Contains ALL experiment ids
experiment_ids_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "metadata",
    "experiment_ids.txt")

normalized_data_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "input",
    "train_set_normalized.pcl")

original_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "Pa_compendium_02.22.2014.pcl")

mapping_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "metadata",
    "sample_annotations.tsv")


# ## Generate simulated data with labels
# 
# Simulate a compendia by experiment and label each new sample with the experiment id that it originated from

# In[5]:


# Load experiment id file
# Contains ALL experiment ids
base_dir = os.path.abspath(
  os.path.join(
      os.getcwd(), "../"))    # base dir on repo

experiment_ids_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "metadata",
    "experiment_ids.txt")


# In[6]:


# Generate simulated data
# Generate simulated data
simulated_labeled_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "simulated_data_labeled.txt.xz")
if not Path(simulated_labeled_data_file).exists():
    generate_labeled_data.simulate_compendium_labeled(experiment_ids_file, 
                                                      num_simulated_experiments,
                                                      normalized_data_file,
                                                      NN_architecture,
                                                      dataset_name,
                                                      local_dir,
                                                      base_dir)


# ## Process data
# 
# Notice: Originally the expression data was 0-1 normalized for use in training the VAE, however when we performed differential expression analyses we found that the normalized data had reduced variance that resulted in an inconsistency between the number of DEGs found compared to the publication. Therefore, we are re-scaling our normalized data to be in the original range of data.

# In[7]:


# Load simulated data
simulated_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "simulated_data_labeled.txt.xz")


# In[8]:


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


# In[9]:


original_data.head(5)


# In[10]:


simulated_data.head(5)


# In[11]:


# 0-1 normalize per gene
scaler = preprocessing.MinMaxScaler()
original_data_scaled = scaler.fit_transform(original_data)
original_data_scaled_df = pd.DataFrame(original_data_scaled,
                                columns=original_data.columns,
                                index=original_data.index)

original_data_scaled_df.head(5)


# In[12]:


# Re-scale simulated data back into the same range as the original data
simulated_data_numeric = simulated_data.drop(columns=['experiment_id'])
simulated_data_scaled = scaler.inverse_transform(simulated_data_numeric)

simulated_data_scaled_df = pd.DataFrame(simulated_data_scaled,
                                columns=simulated_data_numeric.columns,
                                index=simulated_data_numeric.index)

simulated_data_scaled_df['experiment_id'] = simulated_data['experiment_id']
simulated_data_scaled_df.head(5)


# In[13]:


# Read in metadata
metadata = pd.read_table(
    mapping_file, 
    header=0, 
    sep='\t', 
    index_col=0)

metadata.head()


# In[14]:


map_experiment_sample = metadata[['sample_name', 'ml_data_source']]
map_experiment_sample.head()


# # Template experiment E-GEOD-51409
# 
# This experiment measures the transcriptome of *P. aeruginosa* under two different growth conditions: 28 degrees and 37 degress.

# In[15]:


# Get experiment id
experiment_id = 'E-GEOD-51409'


# In[16]:


# Get original samples associated with experiment_id
selected_mapping = map_experiment_sample.loc[experiment_id]
original_selected_sample_ids = list(selected_mapping['ml_data_source'].values)

selected_original_data = original_data.loc[original_selected_sample_ids]

selected_original_data.head(10)


# In[17]:


# Want to get simulated samples associated with experiment_id
# Since we sampled experiments with replacement, we want to find the first set of samples matching the experiment id
match_experiment_id = ''
for experiment_name in simulated_data_scaled_df['experiment_id'].values:
    if experiment_name.split("_")[0] == experiment_id:
        match_experiment_id = experiment_name 


# In[18]:


# Get simulated samples associated with experiment_id
selected_simulated_data = simulated_data_scaled_df[simulated_data_scaled_df['experiment_id'] == match_experiment_id]

# Map sample ids from original data to simulated data
selected_simulated_data.index = original_selected_sample_ids
selected_simulated_data = selected_simulated_data.drop(columns=['experiment_id'])

selected_simulated_data.head(5)


# In[19]:


# Save selected samples
# This will be used as input into R script to identify differentially expressed genes
selected_simulated_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "selected_simulated_data_"+experiment_id+"_example.txt")

selected_original_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "selected_original_data_"+experiment_id+"_example.txt")

selected_simulated_data.to_csv(
        selected_simulated_data_file, float_format='%.3f', sep='\t')

selected_original_data.to_csv(
        selected_original_data_file, float_format='%.3f', sep='\t')


# ## Generate multiple simulated experiments 
# 
# Generate different simulated datasets using the same E-GEOD-51409 template experiment and shifting the experiment in the linear space in different directions multiple times

# In[20]:


# Generate multiple simulated datasets
for i in range(num_runs):
    generate_labeled_data.shift_template_experiment(
        normalized_data_file,
        experiment_id,
        NN_architecture,
        dataset_name,
        local_dir,
        base_dir,
        i)

