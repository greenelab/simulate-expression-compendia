
# coding: utf-8

# # Visualize simulated data with and without noise added
# 
# This noetbook shows the how the structure of the gene expression data is affected in a few cases where noise is added

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import ast
import pandas as pd
import numpy as np
import random
from plotnine import (ggplot, 
                      geom_point,
                      labs,
                      aes, 
                      facet_wrap, 
                      scale_colour_manual,
                      guides, 
                      guide_legend, 
                      theme_bw, 
                      theme,  
                      element_text,
                      element_rect,
                      element_line,
                      element_blank,
                      ggsave)

from sklearn.decomposition import PCA
from keras.models import load_model
import umap

import warnings
warnings.filterwarnings(action='ignore')

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# User parameters
dataset_name = "Pseudomonas_analysis"
analysis_name = 'analysis_0'
NN_architecture = 'NN_2500_30'


# In[3]:


# Load data
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../"))    # base dir on repo
local_dir = "/home/alexandra/Documents"                          # base dir on local machine for data storage

NN_dir = base_dir + "/" + dataset_name + "/models/" + NN_architecture
latent_dim = NN_architecture.split('_')[-1]

simulated_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "experiment_simulated",
    analysis_name,
    "Experiment_1_0.txt.xz")

simulated_noisy_data_file = os.path.join(
    local_dir,
    "Data",
    "Batch_effects",
    "experiment_simulated",
    analysis_name,
    "Experiment_5_0.txt.xz")


# In[4]:


# Output files
umap_overlay_file = os.path.join(
    base_dir,
    "results",
    "Pa_umap_clean_vs_5noise.png")


# In[5]:


# Read data
simulated_data = pd.read_table(
    simulated_data_file,
    header=0,
    sep='\t',
    index_col=0)

simulated_noisy_data = pd.read_table(
    simulated_noisy_data_file,
    header=0,
    sep='\t',
    index_col=0)

print(simulated_data.shape)
print(simulated_noisy_data.shape)


# In[6]:


simulated_data.head(10)


# In[7]:


simulated_noisy_data.head(10)


# In[8]:


# Get and save model
model = umap.UMAP(random_state=randomState).fit(simulated_data)

input_data_UMAPencoded = model.transform(simulated_data)
input_data_UMAPencoded_df = pd.DataFrame(data=input_data_UMAPencoded,
                                         index=simulated_data.index,
                                         columns=['1','2'])


# In[9]:


# UMAP embedding of simulated data
simulated_data_UMAPencoded = model.transform(simulated_noisy_data)
simulated_data_UMAPencoded_df = pd.DataFrame(data=simulated_data_UMAPencoded,
                                         index=simulated_noisy_data.index,
                                         columns=['1','2'])


# In[10]:


# Overlay original input vs simulated data

# Add label for input or simulated dataset
input_data_UMAPencoded_df['dataset'] = 'simulated'
simulated_data_UMAPencoded_df['dataset'] = 'noisy simulated'

# Concatenate input and simulated dataframes together
combined_data_df = pd.concat([input_data_UMAPencoded_df, simulated_data_UMAPencoded_df])

# Plot
g_input_sim = ggplot(combined_data_df[combined_data_df['dataset'] == 'simulated'], aes(x='1', y='2'))
g_input_sim += geom_point(color='#cccccc', 
                          alpha=0.3)
g_input_sim += labs(x = "UMAP 1", 
                    y = "UMAP 2", 
                    title = "UMAP of simulated data with and without noise")
g_input_sim += theme_bw()
g_input_sim += theme(
    legend_title_align = "center",
    plot_background=element_rect(fill='white'),
    legend_key=element_rect(fill='white', colour='white'), 
    plot_title=element_text(weight='bold')
)
g_input_sim += geom_point(combined_data_df[combined_data_df['dataset'] == 'noisy simulated'],
                          alpha=0.15, 
                          color='#b3e5fc')

print(g_input_sim)
ggsave(plot = g_input_sim, filename = umap_overlay_file, dpi=500)

