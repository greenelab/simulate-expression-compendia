
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
import seaborn as sns
import rpy2.robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import matplotlib.pyplot as plt
import seaborn as sns

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


# ## Install R libraries

# In[6]:


get_ipython().run_cell_magic('R', '', '# Select 59\n# Run one time\n#if (!requireNamespace("BiocManager", quietly = TRUE))\n#    install.packages("BiocManager")\n#BiocManager::install("limma")\n#BiocManager::install(\'EnhancedVolcano\')\n#devtools::install_github(\'kevinblighe/EnhancedVolcano\')\n#BiocManager::install(\'clusterProfiler\')\n#BiocManager::install("org.Hs.eg.db")')


# ## Differential expression analysis

# In[7]:


get_ipython().run_cell_magic('R', '', "library('limma')")


# **Get differentially expressed genes from template experiment**

# In[8]:


get_ipython().run_cell_magic('R', '-i metadata_file -i project_id -i template_data_file -i local_dir', '\nsource(\'../functions/DE_analysis.R\')\n\nget_DE_stats(metadata_file,\n             project_id, \n             template_data_file,\n             "template",\n             local_dir,\n             "real")')


# **Get differentially expressed genes from each simulated experiment**

# In[9]:


get_ipython().run_cell_magic('R', '-i metadata_file -i project_id -i base_dir -i local_dir -i num_runs -o num_sign_DEGs_simulated', '\nsource(\'../functions/DE_analysis.R\')\n\nnum_sign_DEGs_simulated <- c()\n\nfor (i in 0:(num_runs-1)){\n  simulated_data_file <- paste(local_dir, "pseudo_experiment/selected_simulated_data_", project_id, "_", i, ".txt", sep="")\n  cat(paste("running file: ", simulated_data_file, "...\\n", sep=""))\n  \n  run_output <- get_DE_stats(metadata_file,\n                             project_id, \n                             simulated_data_file,\n                             "simulated",\n                             local_dir,\n                             i)\n  \n  num_sign_DEGs_simulated <- c(num_sign_DEGs_simulated, run_output)\n}\nmedian(num_sign_DEGs_simulated)')


# In[10]:


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

# In[11]:


# Load association statistics for template experiment
template_DE_stats_file = os.path.join(
    local_dir,
    "DE_stats",
    "DE_stats_template_data_"+project_id+"_real.txt")


# In[12]:


template_DE_stats = pd.read_csv(
    template_DE_stats_file,
    header=0,
    sep='\t',
    index_col=0)

template_DEGs = template_DE_stats[template_DE_stats['adj.P.Val']<0.05]
print(template_DEGs.shape)
template_DEGs.head()


# In[13]:


#%%R
#library(EnhancedVolcano)


# In[14]:


#%%R -i project_id -i template_DE_stats_file -i local_dir

#source('../functions/DE_analysis.R')

#create_volcano(template_DE_stats_file,
#               project_id,
#               "adj.P.Val",
#               local_dir)


# ## Gene set enrichment analysis
# 
# **Goal:** To detect modest but coordinated changes in prespecified sets of related genes (i.e. those genes in the same pathway or share the same GO term).
# 
# 1. Ranks all genes based using DE association statistics. In this case we used the p-value scores to rank genes. logFC returned error -- need to look into this.
# 2. An enrichment score (ES) is defined as the maximum distance from the middle of the ranked list. Thus, the enrichment score indicates whether the genes contained in a gene set are clustered towards the beginning or the end of the ranked list (indicating a correlation with change in expression). 
# 3. Estimate the statistical significance of the ES by a phenotypic-based permutation test in order to produce a null distribution for the ES( i.e. scores based on permuted phenotype)
# 
# **Note:** Since there were 0 differentially expressed genes using simulated experiments, we used gene set enrichement analysis instead of over-representation analysis to get ranking of genes

# In[15]:


get_ipython().run_cell_magic('R', '', 'library(clusterProfiler)\nlibrary(org.Hs.eg.db)\nlibrary(DOSE)')


# **Get pathway enrichment for template experiment**

# In[16]:


get_ipython().run_cell_magic('R', '-i template_DE_stats_file  -o enriched_pathways', "\nsource('../functions/GSEA_analysis.R')\n\nenriched_pathways <- find_enriched_pathways(template_DE_stats_file)")


# In[17]:


print(enriched_pathways.shape)
enriched_pathways.head()


# **Get pathway enrichment for simulated experiments**

# In[18]:


get_ipython().run_cell_magic('R', '-i project_id -i local_dir -i num_runs ', '\nsource(\'../functions/GSEA_analysis.R\')\n\nfor (i in 0:(num_runs-1)){\n    simulated_DE_stats_file <- paste(local_dir, \n                                 "DE_stats/DE_stats_simulated_data_", \n                                 project_id,\n                                 "_", \n                                 i,\n                                 ".txt",\n                                 sep="")\n    cat(paste("running file: ", simulated_DE_stats_file, "...\\n", sep=""))\n    \n    enriched_pathways <- find_enriched_pathways(simulated_DE_stats_file)\n    \n    out_file = paste(local_dir, "GSEA_stats/GSEA_simulated_data_", project_id,"_", i, ".txt", sep="")\n    write.table(enriched_pathways, file = out_file, row.names = T, sep = "\\t", quote = F)\n    }')


# ## Statistics

# **Template experiment**

# In[19]:


col_to_rank = 'enrichmentScore'


# In[20]:


# zip GO ID and description to get unique key
enriched_pathways["ID_description"] = enriched_pathways["ID"] +"-"+enriched_pathways["Description"]


# In[21]:


# Get ranks of template experiment
# Rank pathways by highest enrichment score
enriched_pathways['ranking'] = enriched_pathways[col_to_rank].rank(ascending = 0) 
enriched_pathways = enriched_pathways.sort_values(by=col_to_rank, ascending=False)
enriched_pathways


# In[22]:


# Make dictionary {GO ID-description:rank}
template_rank_dict = dict(zip(enriched_pathways["ID_description"], 
                              enriched_pathways["ranking"]
                             ))
template_rank_dict


# **Check**
# 
# Again, we want to compare our ranked pathways found against what was reported in the original publication.
# 
# *The DEX-responsive genes that we identified are primarily implicated in two broad classes: stress response and development (Table 1; Reimand et al. 2007). Comparison to Gene Ontology (GO) categories (Ashburner et al. 2000) revealed that the identified genes are involved in stress response (P = 6 × 10−11), organ development (P = 5 × 10−15), cell differentiation (P = 1 × 10−11), hormone secretion (P = 4 × 10−7), and apoptosis (P = 5 × 10−12).*
# 
# We found pathways that are consistent with what publication found: pathways related to anatomical structure (i.e. vinculin, ) and cell differentiation (i.e. centromeric sister chromatid cohesion), hormone secretion (i.e. somatic hypermutation of immunoglobulin genes)

# **Simulated experiments**

# In[23]:


def FullMergeDict(D1, D2):
    for key, value in D1.items():
        if key in D2:
            if type(value) is dict:
                FullMergeDict(D1[key], D2[key])
            else:
                if type(value) in (int, float, str):
                    D1[key] = [value]
                if type(D2[key]) is list:
                    D1[key].extend(D2[key])
                else:
                    D1[key].append(D2[key])
    for key, value in D2.items():
        if key not in D1:
            D1[key] = value
    return(D1)


# In[24]:


# Get distribution of ranks for simulated experiments
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
    
    # zip GO ID and description to get unique key
    simulated_enriched_pathways["ID_description"] = simulated_enriched_pathways["ID"]+"-"+simulated_enriched_pathways["Description"]
    
    # Get ranks of simulated experiment
    simulated_enriched_pathways['ranking'] = simulated_enriched_pathways[col_to_rank].rank(ascending = 0) 
    simulated_enriched_pathways = simulated_enriched_pathways.sort_values(by=col_to_rank, ascending=False)
    
    if i == 0:
        # Initiate dictionary {GO ID-description:rank}
        simulated_rank_dict = dict(zip(simulated_enriched_pathways["ID_description"], 
                                      simulated_enriched_pathways["ranking"]
                                     ))
    else:
        tmp_simulated_rank_dict = dict(zip(simulated_enriched_pathways["ID_description"], 
                                      simulated_enriched_pathways["ranking"]
                                     ))
        
        simulated_rank_dict = FullMergeDict(simulated_rank_dict, tmp_simulated_rank_dict)


# In[25]:


simulated_rank_dict


# ## Manually examine enriched pathways
# Examine the enriched pathways identified from template experiment -- How are these enriched pathways ranked in the simulated experiments?
# 
# Recall that there were 25 simulated experiments

# In[26]:


# Compare template rank vs median simulated rank 
# Print those pathways where template rank < simulated rank (template is higher ranked than simulated)
# Some of these are pathways that are potentially template-specific

for key, val in template_rank_dict.items():
    if key in simulated_rank_dict.keys():
        diff_rank = val - np.median(simulated_rank_dict[key])
        if diff_rank < 10:
            print(key)
            print("Template rank: ", val)
            print("Median simulated rank: ", np.median(simulated_rank_dict[key]))


# In[27]:


# Compare template rank vs median simulated rank 
# Print those pathways that ranked high in template experiment AND have similar ranks in simulated experiment
for key, val in template_rank_dict.items():
    if key in simulated_rank_dict.keys():
        diff_rank = val - np.median(simulated_rank_dict[key])
        if (val < 50) and (abs(diff_rank) < 10):
            print(key)
            print("Template rank: ", val)
            print("Median simulated rank: ", np.median(simulated_rank_dict[key]))


# There are a few immune pathways that are template-enriched but there are others that look to be involved in cellular maintenancem, which I wouldn't expect to be specific to these experiment.
# 
# For GSEA I am currently using the p-values from the DE analysis to rank genes, should I try using avg gene expression instead..? Maybe p-values don't cluster genes as much..?
# 
# Note: this notebook takes ~1hr to run
# 
# **Check:**
# * Are there template-specific pathways (i.e. pathways specific to Dexamethasone treatment) that we can use as a positive?
# * Are there agnostic pathways that we can use as a negative control?
