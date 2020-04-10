
# coding: utf-8

# # Create pseudo experiments using simulated compendia
# 
# This notebook is a continuation of ```generate_E_GEOD_51409_template_experiment.ipynb```.  This notebook generates new pseudo-experiments using the experiment-preserving approach from the experiment level simulation. In this simulation we are preserving the experiment type but not the actual experiment so the relationship between samples within an experiment are preserved but the genes that are expressed will be different (see module ```simulate_compendium``` in ```functions/generate_data_parallel.py```).

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
from pathlib import Path
import sys
import ast
import pandas as pd
import numpy as np
import seaborn as sns
import random
import glob
from sklearn import preprocessing

sys.path.append("../")
from functions import utils, generate_labeled_data

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


# In[15]:


map_sample_description = metadata[['ml_data_source', 'description']]
map_sample_description.set_index('ml_data_source', inplace=True)
map_sample_description.head()


# # Template experiment E-GEOD-21704
# 
# This experiment measures the transcriptome of WT and ndk1 mutant *P. aeruginosa* in the presence of exposure to H_2O_2 and polymorphonuclear neutrophils (PMNs), which kill *P. aeruginosa*. 
# 
# More information can be found on the [corresponding array express site](https://www.ebi.ac.uk/arrayexpress/experiments/E-GEOD-21704/)
# 
# Since this experiment contains multiple different comparisons, to show the consistency between the orginal and the simulated experiment, we performed a hierarchal clustering of the expression data.

# In[16]:


# Get experiment id
experiment_id = 'E-GEOD-21704'


# In[17]:


# Output files
heatmap_original_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "results",
    "DE_heatmap_original_"+experiment_id+"_example.svg")

heatmap_simulated_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "results",
    "DE_heatmap_simulated_"+experiment_id+"_example.svg")


# In[18]:


# Get original samples associated with experiment_id
selected_mapping = map_experiment_sample.loc[experiment_id]
original_selected_sample_ids = list(selected_mapping['ml_data_source'].values)

selected_original_data = original_data.loc[original_selected_sample_ids]

# Map numeric sample ids to descriptive ids
desc_id = list(map_sample_description.loc[list(selected_original_data.index)]['description'])
selected_original_data.index = desc_id

# downsample columns 
random_subset_genes = random.sample(selected_original_data.columns.tolist(), 50)

selected_original_data = selected_original_data.loc[:,random_subset_genes]
selected_original_data.head(5)


# In[19]:


# Want to get simulated samples associated with experiment_id
# Since we sampled experiments with replacement, we want to find the first set of samples matching the experiment id
match_experiment_id = ''
for experiment_name in simulated_data_scaled_df['experiment_id'].values:
    if experiment_name.split("_")[0] == experiment_id:
        match_experiment_id = experiment_name 


# In[20]:


# Get simulated samples associated with experiment_id
selected_simulated_data = simulated_data_scaled_df[simulated_data_scaled_df['experiment_id'] == match_experiment_id]

# Map sample ids from original data to simulated data
selected_simulated_data.index = original_selected_sample_ids
selected_simulated_data = selected_simulated_data.drop(columns=['experiment_id'])

selected_simulated_data.index = desc_id

selected_simulated_data = selected_simulated_data.loc[:,random_subset_genes]
selected_simulated_data.head(5)


# In[21]:


# Plot original data
sns.set(style="ticks", context="talk")
sns.set(font='sans-serif', font_scale=1.5)
f = sns.clustermap(selected_original_data.T, cmap="viridis")
f.fig.suptitle('Original experiment') 
f.savefig(heatmap_original_file)


# In[22]:


# Plot simulated data
sns.set(style="ticks", context="talk")
sns.set(font='sans-serif', font_scale=1.5)
f = sns.clustermap(selected_simulated_data.T, cmap="viridis")
f.fig.suptitle('Experiment-level simulated experiment')
f.savefig(heatmap_simulated_file)


# # Template experiment E-GEOD-10030
# 
# This experiment measures the transcriptome of biofilm grown on human cells and planktonic *P. aeruginosa* after treated with Tobramycin, an antibiotic. 
# 
# More information can be found on the [corresponding array express site](https://www.ebi.ac.uk/arrayexpress/experiments/E-GEOD-10030/)
# 
# Since this experiment contains multiple different comparisons, to show the consistency between the orginal and the simulated experiment, we performed a hierarchal clustering of the expression data.

# In[23]:


# Get experiment id
experiment_id = 'E-GEOD-10030'


# In[24]:


# Output files
heatmap_original_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "results",
    "DE_heatmap_original_"+experiment_id+"_example.svg")

heatmap_simulated_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "results",
    "DE_heatmap_simulated_"+experiment_id+"_example.svg")


# In[25]:


# Get original samples associated with experiment_id
selected_mapping = map_experiment_sample.loc[experiment_id]
original_selected_sample_ids = list(selected_mapping['ml_data_source'].values)

selected_original_data = original_data.loc[original_selected_sample_ids]

# Map numeric sample ids to descriptive ids
uniq_desc = map_sample_description.loc[list(selected_original_data.index)].drop_duplicates()
desc_id = list(uniq_desc['description'])
selected_original_data.index = desc_id

# downsample columns 
random_subset_genes = random.sample(selected_original_data.columns.tolist(), 50)

selected_original_data = selected_original_data.loc[:,random_subset_genes]
selected_original_data.head(5)


# In[26]:


# Want to get simulated samples associated with experiment_id
# Since we sampled experiments with replacement, we want to find the first set of samples matching the experiment id
match_experiment_id = ''
for experiment_name in simulated_data_scaled_df['experiment_id'].values:
    if experiment_name.split("_")[0] == experiment_id:
        match_experiment_id = experiment_name 


# In[27]:


# Get simulated samples associated with experiment_id
selected_simulated_data = simulated_data_scaled_df[simulated_data_scaled_df['experiment_id'] == match_experiment_id]

# Map sample ids from original data to simulated data
selected_simulated_data.index = original_selected_sample_ids
selected_simulated_data = selected_simulated_data.drop(columns=['experiment_id'])

selected_simulated_data.index = desc_id

selected_simulated_data = selected_simulated_data.loc[:,random_subset_genes]
selected_simulated_data.head(5)


# In[28]:


# Plot original data
sns.set(style="ticks", context="talk")
sns.set(font='sans-serif', font_scale=1.5)
f = sns.clustermap(selected_original_data.T, cmap="viridis")
f.fig.suptitle('Original experiment')
f.savefig(heatmap_original_file)


# In[29]:


# Plot simulated data
sns.set(style="ticks", context="talk")
sns.set(font='sans-serif', font_scale=1.5)
f = sns.clustermap(selected_simulated_data.T, cmap="viridis")
f.fig.suptitle('Experiment-level simulated experiment')
f.savefig(heatmap_simulated_file)

