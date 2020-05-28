
# coding: utf-8

# # Differential expression analysis
# This notebook performs differential expression analysis using the real template experiment and simulated experiments, as a null set. 

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
dataset_name = params['dataset_name']
num_runs = params['num_simulated']
project_id = params['project_id']

rerun_template = False
rerun_simulated = False


# In[4]:


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
    project_id+"_groups.tsv")


# ## Install R libraries

# In[5]:


get_ipython().run_cell_magic('R', '', '# Select 59\n# Run one time\n#if (!requireNamespace("BiocManager", quietly = TRUE))\n#    install.packages("BiocManager")\n#BiocManager::install("limma")\n#BiocManager::install(\'EnhancedVolcano\')\n#devtools::install_github(\'kevinblighe/EnhancedVolcano\')')


# In[6]:


get_ipython().run_cell_magic('R', '', "library('limma')")


# ### Get differentially expressed genes from template experiment

# In[7]:


get_ipython().run_cell_magic('R', '-i metadata_file -i project_id -i template_data_file -i local_dir -i rerun_template', '\nsource(\'../functions/DE_analysis.R\')\n\nout_file <- paste(local_dir,\n                  "DE_stats/DE_stats_template_data_",\n                  project_id,\n                  "_real.txt",\n                  sep="")\n\nif (rerun_template){\n    get_DE_stats(metadata_file,\n                 project_id, \n                 template_data_file,\n                 "template",\n                 local_dir,\n                 "real")\n    }')


# ### Check signal strength

# In[8]:


# Load association statistics for template experiment
template_DE_stats_file = os.path.join(
    local_dir,
    "DE_stats",
    "DE_stats_template_data_"+project_id+"_real.txt")


# In[9]:


template_DE_stats = pd.read_csv(
    template_DE_stats_file,
    header=0,
    sep='\t',
    index_col=0)

template_DEGs = template_DE_stats[(template_DE_stats['adj.P.Val']<0.001) & 
                                  (template_DE_stats['logFC'].abs()>1)]
print(template_DEGs.shape)
template_DEGs.head(10)


# In[10]:


get_ipython().run_cell_magic('R', '', 'library(EnhancedVolcano)')


# In[11]:


get_ipython().run_cell_magic('R', '-i project_id -i template_DE_stats_file -i local_dir', '\nsource(\'../functions/DE_analysis.R\')\n\ncreate_volcano(template_DE_stats_file,\n               project_id,\n               "adj.P.Val",\n               local_dir)')


# ### Get differentially expressed genes from each simulated experiment

# In[12]:


get_ipython().run_cell_magic('R', '-i metadata_file -i project_id -i base_dir -i local_dir -i num_runs -i rerun_simulated -o num_sign_DEGs_simulated', '\nsource(\'../functions/DE_analysis.R\')\n\nnum_sign_DEGs_simulated <- c()\n\nfor (i in 0:(num_runs-1)){\n    simulated_data_file <- paste(local_dir, \n                                 "pseudo_experiment/selected_simulated_data_",\n                                 project_id,\n                                 "_", \n                                 i,\n                                 ".txt",\n                                 sep="")\n    out_file <- paste(local_dir, \n                      "DE_stats/DE_stats_simulated_data_",\n                      project_id,\n                      "_",\n                      i,\n                      ".txt", \n                      sep="")\n    \n    if (rerun_simulated){\n        run_output <- get_DE_stats(metadata_file,\n                                   project_id, \n                                   simulated_data_file,\n                                   "simulated",\n                                   local_dir,\n                                   i)\n        num_sign_DEGs_simulated <- c(num_sign_DEGs_simulated, run_output)\n    } else {\n        # Read in DE stats data\n        DE_stats_data <- as.data.frame(read.table(out_file, sep="\\t", header=TRUE, row.names=1))\n        \n        # Get number of genes that exceed threshold\n        threshold <- 0.001\n        sign_DEGs <- DE_stats_data[DE_stats_data[,\'adj.P.Val\']<threshold & abs(DE_stats_data[,\'logFC\'])>1,]\n        \n        num_sign_DEGs <- nrow(sign_DEGs)\n        \n        num_sign_DEGs_simulated <- c(num_sign_DEGs_simulated, num_sign_DEGs)\n    }\n}')


# In[13]:


# Plot distribution of differentially expressed genes for simulated experiments
sns.distplot(num_sign_DEGs_simulated,
            kde=False)


# **Observation:** All simulated experiments found 0 DEGs using adjusted p-value cutoff of <5%

# **Check**
# 
# As a check, we compared the number of DEGs identified here versus what was reported in the [Kim et. al. publication](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3566005/#pone.0055596.s008), which found:
# * Four conditions needed to be met for the genes to be selected as differentially expressed genes (DEGs): (i) overall differential expression from the edgeR analysis with FDR < 0.001, (ii) a minimum of 3 patients with significant differential expression, as tested by edgeR for individual differential expression with FDR < 0.01, (iii) consistent up/down regulation among different patients representing more than a two-fold change, and (iv) significant expression in at least 3 patients to remove genes with large fold changes within the noise expression level (FVKM>2 in either normal or tumor tissue). 
# * In total, we selected 1459 genes (543 upregulated and 916 downregulated in tumors) differentially expressed in female NSCLC never-smoker patients
# * Used edgeR to identify DEGs
# 
# By comparison:
# * Our study found 2623 DEGs using limma and applying FDR < 0.001 
# * Spot checking the genes identified with their list of DEGs from S2, we found the some of the same genes and FC direction was consistent. 
# * Currently we are normalizing read counts [downloaded from recount2](https://bioconductor.org/packages/devel/bioc/vignettes/recount/inst/doc/recount-quickstart.html) using RPKM and piping that through limma to identify DEG (this is legacy code from when we expected microarray input instead of RNA-seq)

# ## Get statistics for differential expression analysis

# In[14]:


col_to_rank = 't'


# In[15]:


# Get ranks of template experiment

# If ranking by p-value or adjusted p-value then high rank = low value
if col_to_rank in ['P.Value', 'adj.P.Val']:
    template_DE_stats['ranking'] = template_DE_stats[col_to_rank].rank(ascending = False)
    template_DE_stats = template_DE_stats.sort_values(by=col_to_rank, ascending=True)

# If ranking by logFC then high rank = high abs(value)
elif col_to_rank in ['logFC','t']:
    template_DE_stats['ranking'] = template_DE_stats[col_to_rank].abs().rank(ascending = True)
    template_DE_stats = template_DE_stats.sort_values(by=col_to_rank, ascending=False)

# If ranking by Z-score then high rank = high value
else:
    template_DE_stats['ranking'] = template_DE_stats[col_to_rank].rank(ascending = True)
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
    # If ranking by p-value or adjusted p-value then high rank = low value
    if col_to_rank in ['P.Value', 'adj.P.Val']:
        simulated_DE_stats['ranking'] = simulated_DE_stats[col_to_rank].rank(ascending = False)
        simulated_DE_stats = simulated_DE_stats.sort_values(by=col_to_rank, ascending=True)

    # If ranking by logFC then high rank = high abs(value)
    elif col_to_rank in ['logFC','t']:
        simulated_DE_stats['ranking'] = simulated_DE_stats[col_to_rank].abs().rank(ascending = True)
        simulated_DE_stats = simulated_DE_stats.sort_values(by=col_to_rank, ascending=False)

    # If ranking by Z-score then high rank = high value
    else:
        simulated_DE_stats['ranking'] = simulated_DE_stats[col_to_rank].rank(ascending = True)
        simulated_DE_stats = simulated_DE_stats.sort_values(by=col_to_rank, ascending=False)
    
    # Concatenate df
    simulated_DE_stats_all = pd.concat([simulated_DE_stats_all,
                                       simulated_DE_stats])
    
print(simulated_DE_stats_all.shape)
simulated_DE_stats_all.head()


# In[21]:


if col_to_rank == "adj.P.Val":
    simulated_DE_summary_stats = simulated_DE_stats_all.groupby(['index'])[[col_to_rank, 'adj.P.Val', 'ranking']].agg({
        col_to_rank:['mean', 'std','count', 'median'],
        'ranking':['median']
    })
else:
    simulated_DE_summary_stats = simulated_DE_stats_all.groupby(['index'])[[col_to_rank, 'adj.P.Val', 'ranking']].agg({
        col_to_rank:['mean', 'std','count'],
        'adj.P.Val':['median'],
        'ranking':['median']
    })
simulated_DE_summary_stats.head()


# In[22]:


# Merge template statistics with simulated statistics
template_simulated_DE_stats = template_DE_stats.merge(simulated_DE_summary_stats, 
                                                     left_index=True,
                                                     right_index=True)
print(template_simulated_DE_stats.shape)
template_simulated_DE_stats.head()


# In[29]:


sns.distplot(template_simulated_DE_stats[('ranking','median')].values, kde=False)


# In[24]:


# Parse columns
median_pval_simulated = template_simulated_DE_stats[('adj.P.Val','median')]
median_rank_simulated = template_simulated_DE_stats[('ranking','median')]
mean_test_simulated = template_simulated_DE_stats[(col_to_rank,'mean')]
std_test_simulated = template_simulated_DE_stats[(col_to_rank,'std')]
count_simulated = template_simulated_DE_stats[(col_to_rank,'count')]


# In[25]:


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


# In[26]:


summary.sort_values(by="Z score", ascending=False)


# In[27]:


# Save file
summary_file = os.path.join(
        local_dir, 
        "gene_summary_table.tsv")

summary.to_csv(summary_file, float_format='%.5f', sep='\t')

