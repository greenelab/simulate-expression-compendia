
# coding: utf-8

# # Statistical analysis
# This notebook performs differential expression analysis using the real template experiment and simulated experiments, as a null set. Then the set of differentially expressed genes (DEGs) obtained from this analysis are used to perform gene set enrichment analysis (GSEA) to identify pathways enriched in these set of DEGs.

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


# User selected experiment id
project_id = "SRP000762"


# In[4]:


# Load params
local_dir = params["local_dir"]
dataset_name = params['dataset_name']
num_runs=25


# In[5]:


# Load real template experiment
template_data_file = os.path.join(
    local_dir,
    "recount2_template_data.tsv")

# Load metadata file with grouping assignments for samples
metadata_file = os.path.join(
    base_dir,
    "Rank_pathways",
    "data",
    "metadata",
    "SRP000762_groups.tsv")


# ## Differential expression analysis

# In[6]:


get_ipython().run_cell_magic('R', '', '# Select 59\n# Run one time\n#if (!requireNamespace("BiocManager", quietly = TRUE))\n#    install.packages("BiocManager")\n#BiocManager::install("limma")')


# In[7]:


get_ipython().run_cell_magic('R', '', "library('limma')")


# **Get differentially expressed genes from template experiment**

# In[10]:


get_ipython().run_cell_magic('R', '-i metadata_file -i project_id -i template_data_file -i local_dir', '\nsource(\'../functions/DE_analysis.R\')\n\nget_DE_stats(metadata_file,\n             project_id, \n             template_data_file,\n             "template",\n             local_dir,\n             "real")')


# **Get differentially expressed genes from each simulated experiment**

# In[12]:


get_ipython().run_cell_magic('R', '-i metadata_file -i project_id -i base_dir -i local_dir -i num_runs', '\nsource(\'../functions/DE_analysis.R\')\n\nnum_sign_DEGs_simulated <- c()\n\nfor (i in 0:(num_runs-1)){\n  simulated_data_file <- paste(local_dir, "pseudo_experiment/selected_simulated_data_", project_id, "_", i, ".txt", sep="")\n  cat(paste("running file: ", simulated_data_file, "...\\n", sep=""))\n  \n  run_output <- get_DE_stats(metadata_file,\n                             project_id, \n                             simulated_data_file,\n                             "simulated",\n                             local_dir,\n                             i)\n  \n  num_sign_DEGs_simulated <- c(num_sign_DEGs_simulated, run_output)\n}\nmedian(num_sign_DEGs_simulated)')


# **Check**
# 
# As a check, we compared the number of DEGs identified here versus what was reported in the [Reddy et. al. publication](https://www.ncbi.nlm.nih.gov//pubmed/19801529), which found:
# * 234 genes with a significant (FDR < 5%) change in expression in response to DEX treatment. 
# * After removing pseudogenes (listed in Supplemental Table S3), 209 differentially expressed genes remained 
# * Of the DEX-responsive genes, more showed increases in transcript levels 123 (59%) than showed decreases 86 (41%), and the up-regulation was slightly but significantly stronger than the down-regulation
# 
# By comparison:
# * Our study found 60 DEGs instead of 234. 
# * Spot checking the genes identified with their list of DEX-responsive genes (Supplementary Dataset 2), we found the same genes and FC direction was consistent though magnitudes of theirs was lower compared to ours. 

# In[21]:


# Load association statistics for template experiment
template_DE_stats_file = os.path.join(
    local_dir,
    "DE_stats",
    "DE_stats_template_data_"+project_id+"_real.txt")


# In[35]:


template_DE_stats = pd.read_csv(
    template_DE_stats_file,
    header=0,
    sep='\t',
    index_col=0)

print(template_DE_stats[template_DE_stats['adj.P.Val']<0.05].shape)
template_DE_stats[template_DE_stats['adj.P.Val']<0.05]


# In[23]:


#%%R
#BiocManager::install('EnhancedVolcano')
#devtools::install_github('kevinblighe/EnhancedVolcano')


# In[24]:


get_ipython().run_cell_magic('R', '', 'library(EnhancedVolcano)')


# In[26]:


get_ipython().run_cell_magic('R', '-i project_id -i template_DE_stats_file -i local_dir', '\nsource(\'../functions/DE_analysis.R\')\n\ncreate_volcano(template_DE_stats_file,\n               project_id,\n               "adj.P.Val",\n               local_dir)')


# ## Gene set enrichment analysis
# Use DE association statistics to rank pathways that are enriched....

# **Check**
# 
# Again, we want to compare our ranked pathways found against what was reported in the original publication.
# 
# *The DEX-responsive genes that we identified are primarily implicated in two broad classes: stress response and development (Table 1; Reimand et al. 2007). Comparison to Gene Ontology (GO) categories (Ashburner et al. 2000) revealed that the identified genes are involved in stress response (P = 6 × 10−11), organ development (P = 5 × 10−15), cell differentiation (P = 1 × 10−11), hormone secretion (P = 4 × 10−7), and apoptosis (P = 5 × 10−12).*
