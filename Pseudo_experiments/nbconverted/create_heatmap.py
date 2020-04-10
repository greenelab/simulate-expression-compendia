
# coding: utf-8

# ## Visualize heatmaps of differentially expressed genes
# 
# Visualize a heatmap of the differentially expressed genes using the original E-GEOD-51409 expression data versus the simulated expression data for the same experiment

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
from functions import utils, generate_labeled_data

import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

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
experiment_id = 'E-GEOD-51409'

base_dir = os.path.abspath(
  os.path.join(
      os.getcwd(), "../"))


# In[4]:


# Input files
# File containing expression data from template experiment
selected_original_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "selected_original_data_"+experiment_id+"_example.txt")

selected_simulated_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "selected_simulated_data_"+experiment_id+"_example.txt")

selected_control_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "selected_control_data_"+experiment_id+"_example.txt")

# Files containing DE summary statistics
DE_stats_original_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "output_original",
    "DE_stats_original_data_"+experiment_id+"_example.txt")

DE_stats_simulated_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "output_simulated",
    "DE_stats_simulated_data_"+experiment_id+"_example.txt")

DE_stats_control_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "output_control",
    "DE_stats_control_data_"+experiment_id+"_example.txt")

# Gene number to gene name file
gene_name_file = os.path.join(
    base_dir,
    "Pseudo_experiments",
    "Pseudomonas_aeruginosa_PAO1_107.csv")


# In[5]:


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

heatmap_control_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "results",
    "DE_heatmap_control_"+experiment_id+"_example.svg")

original_sign_DEG_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "results",
    "sign_DEG_original_"+experiment_id+"_example.txt")

simulated_sign_DEG_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "results",
    "sign_DEG_simulated_"+experiment_id+"_example.txt")

control_sign_DEG_file = os.path.join(
    base_dir,
    "Pseudomonas",
    "results",
    "sign_DEG_control_"+experiment_id+"_example.txt")


# In[6]:


# Read data
selected_original_data = pd.read_table(
    selected_original_data_file,
    header=0,
    sep='\t',
    index_col=0)

selected_simulated_data = pd.read_table(
    selected_simulated_data_file,
    header=0,
    sep='\t',
    index_col=0)

selected_control_data = pd.read_table(
    selected_control_data_file,
    header=0,
    sep='\t',
    index_col=0)


DE_stats_original_data = pd.read_table(
    DE_stats_original_file,
    header=0,
    sep='\t',
    index_col=0)

DE_stats_simulated_data = pd.read_table(
    DE_stats_simulated_file,
    header=0,
    sep='\t',
    index_col=0)

DE_stats_control_data = pd.read_table(
    DE_stats_control_file,
    header=0,
    sep='\t',
    index_col=0)

DE_stats_simulated_data.head()


# In[7]:


DE_stats_control_data.head()


# In[8]:


# Experiment-perserving-simulated experiment
# Get DEGs to display in heatmap
# Get the number of genes that adjusted p-value < 0.05 AND log2FC > 1

sign_DEG_simulated = DE_stats_simulated_data[
    (abs(DE_stats_simulated_data['logFC'])>1) & (DE_stats_simulated_data['adj.P.Val']<0.05)]
print(sign_DEG_simulated.shape[0])

# Sort significant DEGs and select top 30 genes
sign_DEG_simulated.sort_values(by=['adj.P.Val'])
sign_DEG_simulated = sign_DEG_simulated.iloc[0:30,]

sign_DEG_simulated.to_csv(
        simulated_sign_DEG_file, float_format='%.3f', sep='\t')

sign_DEG_simulated.head(10)


# In[9]:


DE_stats_original_data.head()


# In[10]:


# Original experiment
# Get DEGs to display in heatmap
# Get the number of genes that adjusted p-value < 0.05 and log2FC > 1

sign_DEG_original = DE_stats_original_data[
    (abs(DE_stats_original_data['logFC'])>1) & (DE_stats_original_data['adj.P.Val']<0.05)]
print(sign_DEG_original.shape[0])

# Sort significant DEGs and select top 30 genes
sign_DEG_original.sort_values(by=['adj.P.Val'])
sign_DEG_original = sign_DEG_original.iloc[0:14,]


sign_DEG_original.to_csv(
        original_sign_DEG_file, float_format='%.3f', sep='\t')

sign_DEG_original.head(10)


# In[11]:


# Randomly-sampled-simulated experiment
# Get DEGs to display in heatmap
# Get the number of genes that adjusted p-value < 0.05 AND log2FC > 1

sign_DEG_control = DE_stats_control_data[
    (abs(DE_stats_control_data['logFC'])>1) & (DE_stats_control_data['adj.P.Val']<0.05)]
print(sign_DEG_control.shape[0])

if sign_DEG_control.shape[0] == 0:
    # Reset data
    sign_DEG_control = DE_stats_control_data
    
    # Sort significant DEGs and select top 30 genes
    sign_DEG_control.sort_values(by=['adj.P.Val'])
    sign_DEG_control = sign_DEG_control.iloc[0:14,]

sign_DEG_control.to_csv(
        control_sign_DEG_file, float_format='%.3f', sep='\t')

sign_DEG_control.head(10)


# In[12]:


# Get gene ids for significant DEGs
original_gene_ids = list(sign_DEG_original.index)
sim_gene_ids = list(sign_DEG_simulated.index)
control_gene_ids = list(sign_DEG_control.index)


# In[13]:


# Read gene number to name mapping
gene_name_mapping = pd.read_table(
    gene_name_file,
    header=0,
    sep=',',
    index_col=0)

gene_name_mapping = gene_name_mapping[["Locus Tag", "Name"]]

gene_name_mapping.set_index("Locus Tag", inplace=True)
gene_name_mapping.head()


# In[14]:


# Format gene numbers to remove extraneous quotes
gene_number = gene_name_mapping.index
gene_name_mapping.index = gene_number.str.strip("\"")

gene_name_mapping.head()


# In[15]:


# Map gene numbers to names
def get_gene_names(gene_id_list):    
    gene_names = []
    for gene_id in gene_id_list:
        gene_name = gene_name_mapping.loc[gene_id]
        if gene_name.isnull()[0]:
            # If gene name does not exist
            # Use gene number
            gene_names.append(gene_id)
        else:
            gene_names.append(gene_name[0])
    return gene_names


# In[16]:


original_gene_names = get_gene_names(original_gene_ids)
control_gene_names = get_gene_names(control_gene_ids)
sim_gene_names = get_gene_names(sim_gene_ids)

print(original_gene_names)
print(control_gene_names)
print(sim_gene_ids)


# In[17]:


# Plot original data
selected_original_DEG_data = selected_original_data[original_gene_ids]
selected_original_DEG_data.columns = original_gene_names
sns.set(style="ticks", context="talk")
sns.set(font='sans-serif', font_scale=1.5)
f = sns.clustermap(selected_original_DEG_data.T, cmap="viridis")
f.fig.suptitle('Original experiment') 
f.savefig(heatmap_original_file)


# In[18]:


# Plot simulated
selected_simulated_DEG_data = selected_simulated_data[sim_gene_ids]
selected_simulated_DEG_data.columns = sim_gene_names
sns.set(font='sans-serif', font_scale=1.5)
f = sns.clustermap(selected_simulated_DEG_data.T, cmap="viridis")
f.fig.suptitle('Experiment-level simulated experiment') 
f.savefig(heatmap_simulated_file)


# In[19]:


# Plot control
selected_control_DEG_data = selected_control_data[control_gene_ids]
selected_control_DEG_data.columns = control_gene_names
sns.set(font='sans-serif', font_scale=1.5)
f = sns.clustermap(selected_control_DEG_data.T, cmap="viridis")
f.fig.suptitle('Sample-level simulated experiment') 
f.savefig(heatmap_control_file)


# **Summary**
# The heatmaps display the top differentially expressed genes (FDR adj p-value < 0.05 and log2 FC > 1) identified using the original expression data (```selected_original_DEG_data```), using simulated expression data created by randomly sampling the compendium (```selected_control_DEG_data```), using the simulated expression data created by sampling by experiment from compendium (```selected_simulated_DEG_data```).
# 
# The heatmaps show that the samples are clustered the same between the original experiment versus the simulated experiment generating using the experiment-preserving approach (```selected_simulated_DEG_data```). This indicates that the VAE, which was used to simulate data, is able to capture and preserve the design of the experiment. However, the there is a difference between the set of DEGs identified in the original versus the simulated experiment, indicating the creation of a “new” experiment.  This new experiment can be used as a hypothesis generating tool - to allow us to explore novel untested experimental stimuli
# 
# The heatmaps also show that the sample clustering is *not* consistent between the simulated experiment generated using the experiment preserving approach (```selected_simulated_DEG_data``` generated by ```generate_E_GEOD_51409_template_experiment.ipynb```) and the simulated experiment generated using the random sampling approach (```selected_control_DEG_data``` generated by ```generate_random_sampled_experiment.ipynb```). This indicates that our added complexity of simulating at the experiment-level compared to the sample-level is more representative of true expression data.
#  
