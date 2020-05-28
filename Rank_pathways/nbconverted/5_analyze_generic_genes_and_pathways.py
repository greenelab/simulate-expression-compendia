
# coding: utf-8

# # Analyze generic genes and pathways
# 
# This notebook uses the DEG and GSEA statistics obtained from the previous notebooks [3_gene_DE_analysis](3_gene_DE_analysis.ipynb) and [4_pathway enrichment analysis](4_pathway_enrichment_analysis.ipynb) to: 
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
import seaborn as sns

from plotnine import (ggplot,
                      labs,   
                      geom_point,
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

# ### Map gene ids
# Our gene ids are ensembl while the published gene ids are using hgnc symbols. We need to map ensembl to hgnc ids in order to compare results.

# In[5]:


get_ipython().run_cell_magic('R', '', 'suppressWarnings(library("biomaRt"))')


# In[6]:


get_ipython().run_cell_magic('R', '-i gene_summary_file -o gene_id_mapping', "# Convert gene ids from ensembl (ours) to entrez (DE_prior)\n\nsource('../functions/GSEA_analysis.R')\n\ngene_id_mapping <- get_ensembl_symbol_mapping(gene_summary_file)")


# In[7]:


# Set ensembl id as index
gene_id_mapping.set_index("ensembl_gene_id", inplace=True)
print(gene_id_mapping.shape)
gene_id_mapping.head()


# In[8]:


# Replace ensembl ids with gene symbols
# Only replace if ensembl ids exist
if gene_id_mapping.shape[0] > 0:
    utils.replace_ensembl_ids(gene_summary_file,
                          gene_id_mapping)


# ### Our DEGs
# Genes are ranked by their adjusted p-value and the median rank reported across 25 simulated experiments is shown in column `Median rank (simulated)`.

# In[9]:


# Read data
gene_stats = pd.read_csv(
    gene_summary_file,
    header=0,
    sep='\t',
    index_col=0)
print(gene_stats.shape)
gene_stats.head()


# In[11]:


# Get list of our genes
gene_ids = list(gene_stats.index)


# ### Published DEGs
# These DEGs are based on the [Crow et. al. publication](https://www.pnas.org/content/pnas/116/13/6491.full.pdf). Their genes are ranked 0 = not commonly DE; 1 = commonly DE. Genes by the number differentially expressed gene sets they appear in and then ranking genes by this score.

# In[12]:


# Get generic genes identified by Crow et. al.
DE_prior_file = "https://raw.githubusercontent.com/maggiecrow/DEprior/master/DE_Prior.txt"

DE_prior = pd.read_csv(DE_prior_file,
                       header=0,
                       sep="\t")

DE_prior.head()


# In[13]:


# Get list of published generic genes
published_generic_genes = list(DE_prior['Gene_Name'])


# ### Compare DEG ranks

# In[15]:


# Get intersection of gene lists
shared_genes = set(gene_ids).intersection(published_generic_genes)
print(len(shared_genes))


# In[25]:


# Get genes only in ours not theirs
our_unique_genes = set(gene_ids) - set(shared_genes)
print(len(our_unique_genes))
our_unique_genes


# In[16]:


# Get rank of intersect genes
our_gene_rank_df = pd.DataFrame(gene_stats.loc[shared_genes,'Median rank (simulated)'])
print(our_gene_rank_df.shape)
our_gene_rank_df.head()


# In[17]:


# Merge published ranking
shared_gene_rank_df = pd.merge(our_gene_rank_df,
                                        DE_prior[['DE_Prior_Rank','Gene_Name']],
                                        left_index=True,
                                        right_on='Gene_Name')

shared_gene_rank_df.set_index('Gene_Name', inplace=True)
print(shared_gene_rank_df.shape)
shared_gene_rank_df.head()


# In[19]:


sns.jointplot(data=shared_gene_rank_df,
              x='Median rank (simulated)',
              y='DE_Prior_Rank',
             kind='hex')

# Make prettier if better way to show it


# In[20]:


# Plot our rank vs their rank on shared genes
fig = ggplot(shared_gene_rank_df, aes(x='Median rank (simulated)', y='DE_Prior_Rank'))
fig += geom_point()
fig += labs(x ='Our Rank',
            y = 'Published Rank',
            title = 'Ranking of generic genes')
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

print(fig)


# ## Generic pathways

# In[21]:


"""
# Read data
pathway_stats = pd.read_csv(
    pathway_summary_file,
    header=0,
    sep='\t',
    index_col=0)

pathway_stats.head()"""


# In[22]:


"""# Define what are the set of generic genes
generic_pathway_data = pathway_stats.sort_values(by="Z score", ascending=True)[0:10]

generic_pathway_data.head()"""


# In[23]:


# Manually compare against Powers et. al publication 
# https://academic.oup.com/bioinformatics/article/34/13/i555/5045793

