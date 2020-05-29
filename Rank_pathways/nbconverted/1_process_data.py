
# coding: utf-8

# # Process data
# This notebook does the following:
# 
# 1. Selects template experiment
# 2. Downloads subset of recount2 data, including the template experiment (50 random experiments + 1 template experiment)
# 3. Train VAE on subset of recount2 data

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import pandas as pd
import numpy as np
import random
import rpy2
import seaborn as sns
from sklearn import preprocessing
import pickle

sys.path.append("../")
from functions import generate_labeled_data, utils, pipeline

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Read in config variables
base_dir = os.path.abspath(os.path.join(os.getcwd(),"../"))

config_file = os.path.abspath(os.path.join(base_dir,
                                           "Rank_pathways",
                                           "init_config.tsv"))
params = utils.read_config(config_file)


# ### Select template experiment
# 
# We manually selected bioproject [SRP012656](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37764), which contains primary non-small cell lung adenocarcinoma tumors and adjacent normal tissues of 6 never-smoker Korean female patients with 2 replicates each.

# In[3]:


# Load params
local_dir = params["local_dir"]
dataset_name = params['dataset_name']
NN_architecture = params['NN_architecture']
project_id = params['project_id']


# ### Download subset of recount2 to use as a compendium
# The compendium will be composed of random experiments + the selected template experiment

# In[4]:


get_ipython().run_cell_magic('R', '', '# Select 59\n# Run one time\n#if (!requireNamespace("BiocManager", quietly = TRUE))\n#    install.packages("BiocManager")\nBiocManager::install("recount")')


# In[5]:


get_ipython().run_cell_magic('R', '', "library('recount')")


# In[6]:


#%%R -i project_id -i base_dir -i local_dir

#source('../functions/download_recount2_data.R')

#get_recount2_compendium(project_id, base_dir, local_dir)


# ### Download expression data for selected project id

# In[7]:


get_ipython().run_cell_magic('R', '-i project_id -i local_dir', "\nsource('../functions/download_recount2_data.R')\n\nget_recount2_template_experiment(project_id, local_dir)")


# ### Subset genes
# For our downstream we will be comparing our set of differentially expression genes against the set found in [Crow et. al. publication](https://www.pnas.org/content/pnas/116/13/6491.full.pdf), we will limit our genes to include only those genes shared between our starting set of genes and those in publication. 

# In[8]:


# Get generic genes identified by Crow et. al.
DE_prior_file = "https://raw.githubusercontent.com/maggiecrow/DEprior/master/DE_Prior.txt"

DE_prior = pd.read_csv(DE_prior_file,
                       header=0,
                       sep="\t")

DE_prior.head()


# In[9]:


# Get list of published generic genes
published_generic_genes = list(DE_prior['Gene_Name'])


# In[10]:


# Get list of our genes
# Load real template experiment
template_data_file = os.path.join(
    local_dir,
    "recount2_template_data.tsv")

# Read template data
template_data = pd.read_csv(
    template_data_file,
    header=0,
    sep='\t',
    index_col=0)

gene_ids = list(template_data.columns)


# In[11]:


# Read file mapping ensembl ids to hgnc symbols
gene_id_file = os.path.join(
    local_dir,
    "ensembl_hgnc_mapping.tsv")

gene_id_mapping = pd.read_csv(
    gene_id_file,
    header=0,
    sep='\t',
    index_col=0)


# In[12]:


"""gene_ids_hgnc = {}
for gene_id in gene_ids:
    gene_id_strip = gene_id.split(".")[0]
    if gene_id_strip in list(gene_id_mapping.index):
        if len(gene_id_mapping.loc[gene_id_strip]) > 1:
            gene_ids_hgnc[gene_id] = gene_id_mapping.loc[gene_id_strip].iloc[0][0]
        else:
            gene_ids_hgnc[gene_id] = gene_id_mapping.loc[gene_id_strip][0]

gene_ids_hgnc"""


# In[13]:


# Save scaler transform
gene_id_dict_file = os.path.join(
    local_dir,
    "gene_id_dict.pickle")
"""
outfile = open(gene_id_dict_file,'wb')
pickle.dump(gene_ids_hgnc,outfile)
outfile.close()"""


# In[14]:


# Load pickled files
gene_ids_hgnc = pickle.load(open(gene_id_dict_file, "rb" ))


# In[15]:


# Get intersection of gene lists
shared_genes_hgnc = set(gene_ids_hgnc.values()).intersection(published_generic_genes)
print(len(shared_genes_hgnc))


# In[16]:


# Convert shared gene ids back to ensembl ids
shared_genes = []
for gene_ensembl, gene_hgnc in gene_ids_hgnc.items():
    if gene_hgnc in shared_genes_hgnc:
        shared_genes.append(gene_ensembl)


# In[27]:


# Save shared genes
shared_genes_file = os.path.join(
    local_dir,
    "shared_gene_ids.pickle")

outfile = open(shared_genes_file,'wb')
pickle.dump(shared_genes,outfile)
outfile.close()


# Since this experiment contains both RNA-seq and smRNA-seq samples which are in different ranges so we will drop smRNA samples so that samples are within the same range. The analysis identifying these two subsets of samples can be found in this [notebook](0_explore_input_data.ipynb)

# In[17]:


# Drop smRNA samples so that samples are within the same range
smRNA_samples = ["SRR493961",
                 "SRR493962",
                 "SRR493963",
                 "SRR493964",
                 "SRR493965",
                 "SRR493966",
                 "SRR493967",
                 "SRR493968",
                 "SRR493969",
                 "SRR493970",
                 "SRR493971",
                 "SRR493972"]


# In[18]:


# Drop samples
template_data = template_data.drop(smRNA_samples)


# In[19]:


# Drop genes
template_data = template_data[shared_genes]

print(template_data.shape)
template_data.head()


# In[20]:


# Save 
template_data.to_csv(template_data_file, float_format='%.5f', sep='\t')


# ### Normalize compendium 

# In[21]:


# Load real gene expression data
original_compendium_file = os.path.join(
    local_dir,
    "recount2_compedium_data.tsv")


# In[22]:


# Read data
original_compendium = pd.read_table(
    original_compendium_file,
    header=0,
    sep='\t',
    index_col=0)

# Drop genes
original_compendium = original_compendium[shared_genes]

print(original_compendium.shape)
original_compendium.head()


# In[29]:


# 0-1 normalize per gene
scaler = preprocessing.MinMaxScaler()
original_data_scaled = scaler.fit_transform(original_compendium)
original_data_scaled_df = pd.DataFrame(original_data_scaled,
                                columns=original_compendium.columns,
                                index=original_compendium.index)

print(original_data_scaled_df.shape)
original_data_scaled_df.head()


# In[28]:


# Save data
normalized_data_file = os.path.join(
    local_dir,
    "normalized_recount2_compendium_data.tsv")

original_data_scaled_df.to_csv(
    normalized_data_file, float_format='%.3f', sep='\t')

original_compendium.to_csv(
    original_compendium_file, float_format='%.3f', sep='\t')

# Save scaler transform
scaler_file = os.path.join(
    local_dir,
    "scaler_transform.pickle")

outfile = open(scaler_file,'wb')
pickle.dump(scaler,outfile)
outfile.close()


# ### Train VAE 

# In[25]:


# Setup directories
# Create VAE directories
output_dirs = [os.path.join(base_dir, dataset_name, "models"),
               os.path.join(base_dir, dataset_name, "logs")]

# Check if analysis output directory exist otherwise create
for each_dir in output_dirs:
    if os.path.exists(each_dir) == False:
        print('creating new directory: {}'.format(each_dir))
        os.makedirs(each_dir, exist_ok=True)

# Check if NN architecture directory exist otherwise create
for each_dir in output_dirs:
    new_dir = os.path.join(each_dir, NN_architecture)
    if os.path.exists(new_dir) == False:
        print('creating new directory: {}'.format(new_dir))
        os.makedirs(new_dir, exist_ok=True)


# In[26]:


# Train VAE on new compendium data
# Write out model to rank_pathways directory
pipeline.train_vae(config_file,
                   normalized_data_file)

