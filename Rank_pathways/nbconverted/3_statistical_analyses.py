
# coding: utf-8

# # Statistical analysis
# This notebook performs differential expression analysis using the real template experiment and simulated experiments, as a null set. Then the set of differentially expressed genes (DEGs) obtained from this analysis are used to perform gene set enrichment analysis (GSEA) to identify pathways enriched in these set of DEGs.
# 
# *Note:* To run datatables, need to refresh the window and then run all cells (DO NOT restart and run all, only run all works).

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import pandas as pd
import numpy as np
import random
import seaborn as sns
import rpy2.robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#from jupyter_datatables import init_datatables_mode


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


# User selected experiment id
project_id = "SRP000762"
rerun_template = True
rerun_simulated = True


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


# ## Install R libraries

# In[6]:


get_ipython().run_cell_magic('R', '', '# Select 59\n# Run one time\n#if (!requireNamespace("BiocManager", quietly = TRUE))\n#    install.packages("BiocManager")\n#BiocManager::install("limma")\n#BiocManager::install(\'EnhancedVolcano\')\n#devtools::install_github(\'kevinblighe/EnhancedVolcano\')\n#BiocManager::install(\'clusterProfiler\')\n#BiocManager::install("org.Hs.eg.db")\n#BiocManager::install("biomaRt")')


# ## Differential expression analysis

# In[7]:


get_ipython().run_cell_magic('R', '', "library('limma')")


# ### Get differentially expressed genes from template experiment

# In[8]:


get_ipython().run_cell_magic('R', '-i metadata_file -i project_id -i template_data_file -i local_dir -i rerun_template', '\nsource(\'../functions/DE_analysis.R\')\n\nout_file <- paste(local_dir,\n                  "DE_stats/DE_stats_template_data_",\n                  project_id,\n                  "_real.txt",\n                  sep="")\n\nif (rerun_template){\n    get_DE_stats(metadata_file,\n                 project_id, \n                 template_data_file,\n                 "template",\n                 local_dir,\n                 "real")\n    }')


# ### Check signal strength

# In[9]:


# Load association statistics for template experiment
template_DE_stats_file = os.path.join(
    local_dir,
    "DE_stats",
    "DE_stats_template_data_"+project_id+"_real.txt")


# In[10]:


template_DE_stats = pd.read_csv(
    template_DE_stats_file,
    header=0,
    sep='\t',
    index_col=0)

template_DEGs = template_DE_stats[template_DE_stats['adj.P.Val']<0.05]
print(template_DEGs.shape)
template_DEGs.head()


# In[11]:


#%%R
#library(EnhancedVolcano)


# In[12]:


#%%R -i project_id -i template_DE_stats_file -i local_dir

#source('../functions/DE_analysis.R')

#create_volcano(template_DE_stats_file,
#               project_id,
#               "adj.P.Val",
#               local_dir)


# ### Get differentially expressed genes from each simulated experiment

# In[13]:


get_ipython().run_cell_magic('R', '-i metadata_file -i project_id -i base_dir -i local_dir -i num_runs -i rerun_simulated -o num_sign_DEGs_simulated', '\nsource(\'../functions/DE_analysis.R\')\n\nnum_sign_DEGs_simulated <- c()\n\nfor (i in 0:(num_runs-1)){\n    simulated_data_file <- paste(local_dir, \n                                 "pseudo_experiment/selected_simulated_data_",\n                                 project_id,\n                                 "_", \n                                 i,\n                                 ".txt",\n                                 sep="")\n    out_file <- paste(local_dir, \n                      "DE_stats/DE_stats_simulated_data_",\n                      project_id,\n                      "_",\n                      i,\n                      ".txt", \n                      sep="")\n    \n    if (rerun_simulated){\n        run_output <- get_DE_stats(metadata_file,\n                                   project_id, \n                                   simulated_data_file,\n                                   "simulated",\n                                   local_dir,\n                                   i)\n        num_sign_DEGs_simulated <- c(num_sign_DEGs_simulated, run_output)\n    } else {\n        # Read in DE stats data\n        DE_stats_data <- as.data.frame(read.table(out_file, sep="\\t", header=TRUE, row.names=1))\n        \n        # Get number of genes that exceed threshold\n        threshold <- 0.05\n        sign_DEGs <- DE_stats_data[DE_stats_data[,\'adj.P.Val\']<threshold,]\n        \n        num_sign_DEGs <- nrow(sign_DEGs)\n        \n        num_sign_DEGs_simulated <- c(num_sign_DEGs_simulated, num_sign_DEGs)\n    }\n}')


# In[14]:


# Plot distribution of differentially expressed genes for simulated experiments
sns.distplot(num_sign_DEGs_simulated,
            kde=False)


# **Observation:** All simulated experiments found 0 DEGs using adjusted p-value cutoff of <5%

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

# ## Get statistics for differential expression analysis

# In[15]:


col_to_rank = 'logFC'


# In[16]:


# Get ranks of template experiment
# Rank pathways by highest enrichment score
template_DE_stats['ranking'] = template_DE_stats[col_to_rank].rank(ascending = 0) 
template_DE_stats = template_DE_stats.sort_values(by=col_to_rank, ascending=False)

template_DE_stats.head()


# In[17]:


# Concatenate simulated experiments
simulated_DE_stats_all = pd.DataFrame()
for i in range(num_runs):
    simulated_DE_stats_file = os.path.join(
        local_dir, 
        "DE_stats",
        "DE_stats_simulated_data_"+project_id+"_"+str(i)+".txt")
    
    #Read results
    simulated_DE_stats = pd.read_csv(
        simulated_DE_stats_file,
        header=0,
        sep='\t',
        index_col=0)
    
    simulated_DE_stats.reset_index(inplace=True)
    
    # Add ranks of simulated experiment
    simulated_DE_stats['ranking'] = simulated_DE_stats[col_to_rank].rank(ascending = 0) 
    simulated_DE_stats = simulated_DE_stats.sort_values(by=col_to_rank, ascending=False)
    
    # Concatenate df
    simulated_DE_stats_all = pd.concat([simulated_DE_stats_all,
                                       simulated_DE_stats])
    
print(simulated_DE_stats_all.shape)
simulated_DE_stats_all.head()


# In[18]:


simulated_DE_summary_stats = simulated_DE_stats_all.groupby(['index'])[[col_to_rank, 'adj.P.Val', 'ranking']].agg({
    col_to_rank:['mean', 'std','count'],
    'adj.P.Val':['median'],
    'ranking':['median']
})

simulated_DE_summary_stats.head()


# In[19]:


# Merge template statistics with simulated statistics
template_simulated_DE_stats = template_DE_stats.merge(simulated_DE_summary_stats, 
                                                     left_index=True,
                                                     right_index=True)
print(template_simulated_DE_stats.shape)
template_simulated_DE_stats.head()


# In[20]:


# Parse columns
median_pval_simulated = template_simulated_DE_stats[('adj.P.Val','median')]
median_rank_simulated = template_simulated_DE_stats[('ranking','median')]
mean_test_simulated = template_simulated_DE_stats[(col_to_rank,'mean')]
std_test_simulated = template_simulated_DE_stats[(col_to_rank,'std')]
count_simulated = template_simulated_DE_stats[(col_to_rank,'count')]


# In[22]:


summary = pd.DataFrame(data={'Gene ID': template_simulated_DE_stats.index,
                             'Adj P-value (Real)': template_simulated_DE_stats['adj.P.Val'],
                             'Rank (Real)': template_simulated_DE_stats['ranking'],
                             'Test statistic (Real)': template_simulated_DE_stats[col_to_rank],
                             'Median adj p-value (simulated)': median_pval_simulated ,
                             'Median rank (simulated)': median_rank_simulated ,
                             'Mean test statistic (simulated)': mean_test_simulated ,
                             'Std deviation (simulated)': std_test_simulated,
                             'Number of experiments (simulated)': count_simulated
                            }
                      )
summary['Z score'] = (summary['Test statistic (Real)'] - summary['Mean test statistic (simulated)'])/summary['Std deviation (simulated)']
summary.head()


# In[23]:


summary.sort_values(by="Z score", ascending=False)


# In[24]:


# Save file
summary_file = os.path.join(
        local_dir, 
        "gene_summary_table.tsv")

summary.to_csv(summary_file, float_format='%.5f', sep='\t')


# ## Gene set enrichment analysis
# 
# **Goal:** To detect modest but coordinated changes in prespecified sets of related genes (i.e. those genes in the same pathway or share the same GO term).
# 
# 1. Ranks all genes based using DE association statistics. In this case we used the p-value scores to rank genes. logFC returned error -- need to look into this.
# 2. An enrichment score (ES) is defined as the maximum distance from the middle of the ranked list. Thus, the enrichment score indicates whether the genes contained in a gene set are clustered towards the beginning or the end of the ranked list (indicating a correlation with change in expression). 
# 3. Estimate the statistical significance of the ES by a phenotypic-based permutation test in order to produce a null distribution for the ES( i.e. scores based on permuted phenotype)
# 
# **Note:** Since there were 0 differentially expressed genes using simulated experiments, we used gene set enrichement analysis instead of over-representation analysis to get ranking of genes

# In[25]:


# Load Hallmark pathways database used by Powers et. al.
# https://github.com/CostelloLab/GSEA-InContext/blob/master/data/gene_sets/hallmarks.gmt
hallmark_DB_file = os.path.join(
    local_dir,
    "hallmark_DB.gmt")


# In[26]:


get_ipython().run_cell_magic('R', '', 'suppressWarnings(library("clusterProfiler"))\nsuppressWarnings(library("org.Hs.eg.db"))\nsuppressWarnings(library("DOSE"))\nsuppressWarnings(library("biomaRt"))')


# ### Get pathway enrichment for template experiment

# In[27]:


get_ipython().run_cell_magic('R', '-i template_DE_stats_file -o gene_id_mapping', "\nsource('../functions/GSEA_analysis.R')\n\ngene_id_mapping <- get_ensembl_symbol_mapping(template_DE_stats_file)")


# In[28]:


# Set ensembl id as index
gene_id_mapping.set_index("ensembl_gene_id", inplace=True)
print(gene_id_mapping.shape)
gene_id_mapping.head()


# In[29]:


# Replace ensembl ids with gene symbols
utils.replace_ensembl_ids(template_DE_stats_file,
                          gene_id_mapping)


# In[30]:


get_ipython().run_cell_magic('R', '-i template_DE_stats_file -i hallmark_DB_file -o template_enriched_pathways', "\nsource('../functions/GSEA_analysis.R')\n\ntemplate_enriched_pathways <- find_enriched_pathways(template_DE_stats_file, hallmark_DB_file)")


# In[31]:


print(template_enriched_pathways.shape)
template_enriched_pathways.head()


# ### Get pathway enrichment for simulated experiments

# In[32]:


# Replace ensembl ids with gene symbols
if rerun_simulated:
    for i in range(num_runs):
        simulated_DE_stats_file = os.path.join(
            local_dir, 
            "DE_stats",
            "DE_stats_simulated_data_"+project_id+"_"+str(i)+".txt")

        utils.replace_ensembl_ids(simulated_DE_stats_file,
                                  gene_id_mapping)


# In[33]:


get_ipython().run_cell_magic('R', '-i project_id -i local_dir -i hallmark_DB_file -i num_runs -i rerun_simulated', '\nsource(\'../functions/GSEA_analysis.R\')\n\nfor (i in 0:(num_runs-1)){\n    simulated_DE_stats_file <- paste(local_dir, \n                                 "DE_stats/DE_stats_simulated_data_", \n                                 project_id,\n                                 "_", \n                                 i,\n                                 ".txt",\n                                 sep="")\n    \n    out_file = paste(local_dir, \n                     "GSEA_stats/GSEA_simulated_data_",\n                     project_id,\n                     "_",\n                     i,\n                     ".txt", \n                     sep="")\n    \n    if (rerun_simulated){\n        enriched_pathways <- find_enriched_pathways(simulated_DE_stats_file, hallmark_DB_file) \n    \n        write.table(enriched_pathways, file = out_file, row.names = T, sep = "\\t", quote = F)\n        }\n    }')


# **Check**
# 
# Again, we want to compare our ranked pathways found against what was reported in the original publication.
# 
# *The DEX-responsive genes that we identified are primarily implicated in two broad classes: stress response and development (Table 1; Reimand et al. 2007). Comparison to Gene Ontology (GO) categories (Ashburner et al. 2000) revealed that the identified genes are involved in stress response (P = 6 × 10−11), organ development (P = 5 × 10−15), cell differentiation (P = 1 × 10−11), hormone secretion (P = 4 × 10−7), and apoptosis (P = 5 × 10−12).*
# 
# We found pathways that are consistent with what publication found: pathways related to anatomical structure (i.e. vinculin, ) and cell differentiation (i.e. centromeric sister chromatid cohesion), hormone secretion (i.e. somatic hypermutation of immunoglobulin genes)

# ## Get statistics for enriched pathways
# Examine the enriched pathways identified from template experiment -- How are these enriched pathways ranked in the simulated experiments?

# ### Template experiment

# In[34]:


col_to_rank = 'enrichmentScore'


# In[35]:


# Get ranks of template experiment
# Rank pathways by highest enrichment score
template_enriched_pathways['ranking'] = template_enriched_pathways[col_to_rank].rank(ascending = 0) 
template_enriched_pathways = template_enriched_pathways.sort_values(by=col_to_rank, ascending=False)

# Set index to GO ID
template_enriched_pathways.set_index("ID", inplace=True)
print(template_enriched_pathways.shape)
template_enriched_pathways.head()


# In[36]:


# Check that GO IDs are unique
template_enriched_pathways.index.nunique() == len(template_enriched_pathways)


# ### Simulated experiments 

# In[37]:


# Concatenate simulated experiments
simulated_enriched_pathways_all = pd.DataFrame()
for i in range(num_runs):
    simulated_GSEA_file = os.path.join(
        local_dir, 
        "GSEA_stats",
        "GSEA_simulated_data_"+project_id+"_"+str(i)+".txt")
    
    #Read results
    simulated_enriched_pathways = pd.read_csv(
        simulated_GSEA_file,
        header=0,
        sep='\t',
        index_col=0)
    
    # Add ranks of simulated experiment
    simulated_enriched_pathways['ranking'] = simulated_enriched_pathways[col_to_rank].rank(ascending = 0) 
    simulated_enriched_pathways = simulated_enriched_pathways.sort_values(by=col_to_rank, ascending=False)
    
    # Concatenate df
    simulated_enriched_pathways_all = pd.concat([simulated_enriched_pathways_all,
                                       simulated_enriched_pathways])
    
print(simulated_enriched_pathways_all.shape)
simulated_enriched_pathways_all.head()


# In[38]:


simulated_enriched_pathways_stats = simulated_enriched_pathways_all.groupby(['ID'])[['enrichmentScore', 'pvalue', 'ranking']].agg({
    col_to_rank:['mean', 'std','count'],
    'pvalue':['median'],
    'ranking':['median']
})

simulated_enriched_pathways_stats.head()


# In[39]:


# Merge template statistics with simulated statistics
template_simulated_enriched_pathways_stats = template_enriched_pathways.merge(simulated_enriched_pathways_stats, 
                                                                              how='outer',
                                                                              left_index=True,
                                                                              right_index=True)
template_simulated_enriched_pathways_stats.head()


# In[40]:


# Parse columns
median_pval_simulated = template_simulated_enriched_pathways_stats[('pvalue','median')]
median_rank_simulated = template_simulated_enriched_pathways_stats[('ranking','median')]
mean_test_simulated = template_simulated_enriched_pathways_stats[(col_to_rank,'mean')]
std_test_simulated = template_simulated_enriched_pathways_stats[(col_to_rank,'std')]
count_simulated = template_simulated_enriched_pathways_stats[(col_to_rank,'count')]


# ### Calculations for summary table

# In[41]:


#init_datatables_mode()


# In[42]:


summary = pd.DataFrame(data={'Pathway': template_simulated_enriched_pathways_stats.index,
                             'P-value (Real)': template_simulated_enriched_pathways_stats['pvalue'],
                             'Rank (Real)': template_simulated_enriched_pathways_stats['ranking'],
                             'Test statistic (Real)': template_enriched_pathways[col_to_rank],
                             'Median p-value (simulated)': median_pval_simulated ,
                             'Median rank (simulated)': median_rank_simulated ,
                             'Mean test statistic (simulated)': mean_test_simulated ,
                             'Std deviation (simulated)': std_test_simulated,
                             'Number of experiments (simulated)': count_simulated
                            }
                      )
summary['Z score'] = (summary['Test statistic (Real)'] - summary['Mean test statistic (simulated)'])/summary['Std deviation (simulated)']
summary


# In[43]:


# Save file
summary_file = os.path.join(
        local_dir, 
        "pathway_summary_table.tsv")

summary.to_csv(summary_file, float_format='%.5f', sep='\t')


# There are a few immune pathways that are template-enriched but there are others that look to be involved in cellular maintenancem, which I wouldn't expect to be specific to these experiment.
# 
# For GSEA I am currently using the p-values from the DE analysis to rank genes, should I try using avg gene expression instead..? Maybe p-values don't cluster genes as much..?
# 
# Note: this notebook takes ~1hr to run
# 
# **Check:**
# * Are there template-specific pathways (i.e. pathways specific to Dexamethasone treatment) that we can use as a positive?
# * Are there agnostic pathways that we can use as a negative control?
