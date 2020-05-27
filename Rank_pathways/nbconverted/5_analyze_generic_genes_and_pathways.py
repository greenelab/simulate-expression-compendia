
# coding: utf-8

# # Analyze generic genes and pathways
# 
# This notebook uses the statistics obtained from the [previous notebook](3_statistical_analyses.ipynb) to 
# 1. Determine if our simulation approach can identify a set of generic genes and pathways
# 2. Compare our set of generic genes and pathways with what has been previously reported

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import pandas as pd
import numpy as np
import random
import warnings
import rpy2.robjects

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

sys.path.append("../")
from functions import utils

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


# In[3]:


# Load params
local_dir = params["local_dir"]


# In[4]:


# Input files
gene_summary_file = os.path.join(
    local_dir, 
    "gene_summary_table.tsv")

pathway_summary_file = os.path.join(
    local_dir, 
    "pathway_summary_table.tsv")


# ## Generic genes

# In[8]:


get_ipython().run_cell_magic('R', '', 'suppressWarnings(library("biomaRt"))')


# In[9]:


get_ipython().run_cell_magic('R', '-i gene_summary_file -o gene_id_mapping', "# Convert gene ids from ensembl (ours) to entrez (DE_prior)\n\nsource('../functions/GSEA_analysis.R')\n\ngene_id_mapping <- get_ensembl_symbol_mapping(gene_summary_file)")


# In[10]:


# Set ensembl id as index
gene_id_mapping.set_index("ensembl_gene_id", inplace=True)
print(gene_id_mapping.shape)
gene_id_mapping.head()


# In[12]:


# Replace ensembl ids with gene symbols
utils.replace_ensembl_ids(gene_summary_file,
                          gene_id_mapping)


# In[13]:


# Read data
gene_stats = pd.read_csv(
    gene_summary_file,
    header=0,
    sep='\t',
    index_col=0)

gene_stats.head()


# In[14]:


# Define what are the set of generic genes
generic_genes_data = gene_stats.sort_values(by="Z score", ascending=True)[0:10]

generic_genes_data.head()


# In[15]:


# Get list of generic genes
generic_genes = list(generic_genes_data.index)


# In[30]:


# Get generic genes identified by Crow et. al.
# https://www.pnas.org/content/pnas/116/13/6491.full.pdf
DE_prior_file = "https://raw.githubusercontent.com/maggiecrow/DEprior/master/DE_Prior.txt"

DE_prior = pd.read_csv(DE_prior_file,
                       header=0,
                       sep="\t")

DE_prior.head()


# In[38]:


# Get list of published generic genes
published_generic_genes = list(DE_prior['Gene_Name'])


# In[45]:


# What is the percent of our genes that intersects with those previously reported?
print(set(published_generic_genes).intersection(generic_genes))
len(set(published_generic_genes).intersection(generic_genes))/len(generic_genes)


# ## Generic pathways

# In[46]:


# Read data
pathway_stats = pd.read_csv(
    pathway_summary_file,
    header=0,
    sep='\t',
    index_col=0)

pathway_stats.head()


# In[47]:


# Define what are the set of generic genes
generic_pathway_data = pathway_stats.sort_values(by="Z score", ascending=True)[0:10]

generic_pathway_data.head()


# In[ ]:


# Manually compare against Powers et. al publication 
# https://academic.oup.com/bioinformatics/article/34/13/i555/5045793

