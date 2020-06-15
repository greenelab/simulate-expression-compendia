
# coding: utf-8

# # Generate and visualize experiment-level simulated data
# 
# The goal of this notebook is to create a simulated compendium, keep track of the relationship between samples and experiments.
# 
# Then visualizing the placement of the original experiment and the simulated experiment
# 
# This figure can be found in the manuscript (Figure 4B)

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import glob
import pandas as pd
import numpy as np
import random
import umap

import warnings
warnings.filterwarnings(action='ignore')

from plotnine import (ggplot,
                      labs,  
                      geom_line, 
                      geom_point,
                      geom_errorbar,
                      aes, 
                      ggsave, 
                      theme_bw,
                      theme,
                      xlim,
                      ylim,
                      facet_wrap,
                      scale_color_manual,
                      guides, 
                      guide_legend,
                      element_blank,
                      element_text,
                      element_rect,
                      element_line,
                      coords)

from ponyo import utils, generate_labeled_data

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Read in config variables
config_file = os.path.abspath(os.path.join(os.getcwd(),"../configs", "config_Pa_experiment_limma.tsv"))
params = utils.read_config(config_file)


# In[3]:


# Load parameters
local_dir = params["local_dir"]
dataset_name = params['dataset_name']
analysis_name = params["simulation_type"]
num_simulated_experiments = params["num_simulated_experiments"]
train_architecture = params['NN_architecture']

base_dir = os.path.abspath(
  os.path.join(
      os.getcwd(), "../"))


# In[4]:


# Input files
normalized_data_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "input",
    "train_set_normalized.pcl")

metadata_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "metadata",
    "sample_annotations.tsv")


# In[5]:


# Output
experiment_simulated_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "results",
    "simulated_experiment_compendia.svg")


# ### Load file with experiment ids

# In[6]:


experiment_ids_file = os.path.join(
    base_dir,
    dataset_name,
    "data",
    "metadata",
    "experiment_ids.txt")


# ### Generate simulated data with experiment ids

# In[7]:


# Generate simulated data
#generate_labeled_data.simulate_compendium_labeled(experiment_ids_file, 
#                                                  num_simulated_experiments,
#                                                  normalized_data_file,
#                                                  NN_architecture,
#                                                  dataset_name,
#                                                  local_dir,
#                                                  base_dir
#                                                 )


# ### Load simulated gene expression data

# In[8]:


# Simulated data file 
simulated_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "simulated_data_labeled.txt.xz")


# In[9]:


# Read in simulated data
simulated_data = pd.read_table(
    simulated_data_file,
    header=0,
    index_col=0,
    sep='\t')

simulated_data.head()


# In[10]:


# Number of unique experiments in simulated dataset
ids = set([i.split("_")[0] for i in simulated_data['experiment_id']])
len(ids)     


# ### Add experiment ids to original gene expression data

# In[11]:


# Read original input
normalized_data = pd.read_table(
        normalized_data_file,
        header=0,
        sep='\t',
        index_col=0).T

normalized_data.head()


# In[12]:


# Read in metadata
metadata = pd.read_table(
    metadata_file,
    header=0,
    index_col=0,
    sep='\t')


# In[13]:


# Reset index to be referenced based on sample id
metadata = metadata.reset_index().set_index('ml_data_source')
metadata.head()


# In[14]:


# Remove sample ids that have duplicates
metadata = metadata.loc[~normalized_data.index.duplicated(keep=False)]


# In[15]:


# Add experiment id to original gene expression data
sample_ids = list(normalized_data.index)
normalized_data_label = normalized_data.copy()
for sample_id in sample_ids:
    if sample_id in list(metadata.index):
        if metadata.loc[sample_id].ndim == 1:
            normalized_data_label.loc[sample_id,'experiment_id'] = metadata.loc[sample_id,'experiment']
        else:
            normalized_data_label.loc[sample_id,'experiment_id'] = 'NA'
    else:
        normalized_data_label.loc[sample_id,'experiment_id'] = 'NA'

normalized_data_label.head()


# ## Visualize data

# In[16]:


# Select example experiments
example_id_sim = "E-GEOD-51409_173"
example_id = "E-GEOD-51409"
#example_id_sim = "E-GEOD-52445_58"
#example_id = "E-GEOD-52445"
#example_id_sim = "E-GEOD-18594_2"
#example_id = "E-GEOD-18594"
#example_id_sim = "E-MEXP-2606_0"
#example_id = "E-MEXP-2606"


# In[17]:


# Only label selected example labels for simulated data
simulated_data.loc[simulated_data['experiment_id'] == example_id_sim,'experiment_id'] = example_id

print(example_id in list(simulated_data['experiment_id']))

simulated_data.loc[simulated_data['experiment_id'] != example_id,'experiment_id'] = "not selected"

simulated_data.head()


# In[18]:


example_id in list(simulated_data['experiment_id'])


# In[19]:


# Only label selected example labels for original data
normalized_data_label.loc[normalized_data_label['experiment_id'] == example_id,'experiment_id'] = example_id
normalized_data_label.loc[normalized_data_label['experiment_id'] != example_id,'experiment_id'] = "not selected"

normalized_data_label.head()


# In[20]:


example_id in list(normalized_data_label['experiment_id'])


# In[21]:


# UMAP embedding of original input data

# Get and save model
model = umap.UMAP(random_state=randomState).fit(normalized_data)

input_data_UMAPencoded = model.transform(normalized_data)
input_data_UMAPencoded_df = pd.DataFrame(data=input_data_UMAPencoded,
                                         index=normalized_data.index,
                                         columns=['1','2'])
# Add label
input_data_UMAPencoded_df['experiment_id'] = normalized_data_label['experiment_id']


# In[22]:


# UMAP embedding of simulated data

# Drop label column
simulated_data_numeric = simulated_data.drop(['experiment_id'], axis=1)

simulated_data_UMAPencoded = model.transform(simulated_data_numeric)
simulated_data_UMAPencoded_df = pd.DataFrame(data=simulated_data_UMAPencoded,
                                         index=simulated_data.index,
                                         columns=['1','2'])

# Add back label column
simulated_data_UMAPencoded_df['experiment_id'] = simulated_data['experiment_id']

# downsample backgrd
simulated_data_UMAPencoded_df_backgrd = simulated_data_UMAPencoded_df[
    simulated_data_UMAPencoded_df['experiment_id'] == 'not selected'].sample(n=1000)

simulated_data_UMAPencoded_df_exp = simulated_data_UMAPencoded_df[
    simulated_data_UMAPencoded_df['experiment_id'] != 'not selected']

# Concatenate dataframes
simulated_data_UMAPencoded_df = simulated_data_UMAPencoded_df_backgrd.append(simulated_data_UMAPencoded_df_exp)

simulated_data_UMAPencoded_df.shape


# In[23]:


# Add label for input or simulated dataset
input_data_UMAPencoded_df['dataset'] = 'original'
simulated_data_UMAPencoded_df['dataset'] = 'simulated'

# Concatenate input and simulated dataframes together
combined_data_df = pd.concat([input_data_UMAPencoded_df, simulated_data_UMAPencoded_df])

# Plot
fig = ggplot(combined_data_df, aes(x='1', y='2'))
fig += geom_point(aes(color='experiment_id'), alpha=0.1)
fig += facet_wrap('~dataset')
fig += labs(x ='UMAP 1',
            y = 'UMAP 2',
            title = 'UMAP of original and simulated data (gene space)')
fig += theme_bw()
fig += theme(
    legend_title_align = "center",
    plot_background=element_rect(fill='white'),
    legend_key=element_rect(fill='white', colour='white'), 
    legend_title=element_text(family='sans-serif', size=15),
    legend_text=element_text(family='sans-serif', size=12),
    plot_title=element_text(family='sans-serif', size=15),
    axis_text=element_text(family='sans-serif', size=12),
    axis_title=element_text(family='sans-serif', size=15)
    )
fig += guides(colour=guide_legend(override_aes={'alpha': 1}))
fig += scale_color_manual(['red', '#bdbdbd'])
fig += geom_point(data=combined_data_df[combined_data_df['experiment_id'] == example_id],
                  alpha=0.1, 
                  color='red')

print(fig)
ggsave(plot=fig, filename=experiment_simulated_file)

