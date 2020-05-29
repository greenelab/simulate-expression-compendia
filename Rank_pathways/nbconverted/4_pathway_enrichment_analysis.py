
# coding: utf-8

# # Gene set enrichment analysis
# 
# **Goal:** To detect modest but coordinated changes in prespecified sets of related genes (i.e. those genes in the same pathway or share the same GO term).
# 
# 1. Ranks all genes based using DE association statistics. In this case we used the p-value scores to rank genes. logFC returned error -- need to look into this.
# 2. An enrichment score (ES) is defined as the maximum distance from the middle of the ranked list. Thus, the enrichment score indicates whether the genes contained in a gene set are clustered towards the beginning or the end of the ranked list (indicating a correlation with change in expression). 
# 3. Estimate the statistical significance of the ES by a phenotypic-based permutation test in order to produce a null distribution for the ES( i.e. scores based on permuted phenotype)

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

rerun_template = True
rerun_simulated = False


# ### Install R libraries

# In[4]:


get_ipython().run_cell_magic('R', '', '# Select 59\n# Run one time\n#if (!requireNamespace("BiocManager", quietly = TRUE))\n#    install.packages("BiocManager")\n#BiocManager::install(\'clusterProfiler\')\n#BiocManager::install("biomaRt")\n#install.packages("devtools")\n#BiocManager::install("GSA")')


# In[5]:


get_ipython().run_cell_magic('R', '', 'suppressWarnings(library("clusterProfiler"))\nsuppressWarnings(library("org.Hs.eg.db"))\nsuppressWarnings(library("DOSE"))\nsuppressWarnings(library("biomaRt"))\nsuppressWarnings(library("fgsea"))\nsuppressWarnings(library("GSA"))')


# ### Get pathway enrichment for template experiment

# In[6]:


# Load Hallmark pathways database used by Powers et. al.
# https://github.com/CostelloLab/GSEA-InContext/blob/master/data/gene_sets/hallmarks.gmt
hallmark_DB_file = os.path.join(
    local_dir,
    "hallmark_DB.gmt")


# In[7]:


template_DE_stats_file = os.path.join(
    local_dir,
    "DE_stats",
    "DE_stats_template_data_"+project_id+"_real.txt")


# In[8]:


get_ipython().run_cell_magic('R', '-i template_DE_stats_file -o gene_id_mapping', "\nsource('../functions/GSEA_analysis.R')\n\ngene_id_mapping <- get_ensembl_symbol_mapping(template_DE_stats_file)")


# In[9]:


# Set ensembl id as index
gene_id_mapping.set_index("ensembl_gene_id", inplace=True)
print(gene_id_mapping.shape)
gene_id_mapping.head()


# In[10]:


# Replace ensembl ids with gene symbols
utils.replace_ensembl_ids(template_DE_stats_file,
                          gene_id_mapping)


# In[11]:


# Shuffle ranking
# Create 5 shuffled datasets
for r in range(5):
    template_DE_stats = pd.read_csv(
    template_DE_stats_file,
    header=0,
    sep='\t',
    index_col=0)
    
    shuffled_template_DE_stats_file = os.path.join(
    local_dir,
    "DE_stats",
    "DE_stats_template_data_"+project_id+"_shuffled_"+str(r)+".txt")
    
    shuffled_template_DE_stats = template_DE_stats.iloc[np.random.permutation(len(template_DE_stats))]
    shuffled_template_DE_stats.index = template_DE_stats.index
    
    # Save 
    shuffled_template_DE_stats.to_csv(shuffled_template_DE_stats_file, float_format='%.5f', sep='\t')


# In[12]:


# Test shuffled files
r = 2
shuffled_template_DE_stats_file = os.path.join(
    local_dir,
    "DE_stats",
    "DE_stats_template_data_"+project_id+"_shuffled_"+str(r)+".txt")


# In[13]:


#%%R
#wpgmtfile <- system.file("extdata/wikipathways-20180810-gmt-Homo_sapiens.gmt", package="clusterProfiler")
#wp2gene <- read.gmt(wpgmtfile)
#wpid2gene <- wp2gene %>% dplyr::select(wpid, gene) #TERM2GENE
#wp2gene


# In[24]:


get_ipython().run_cell_magic('R', '-i template_DE_stats_file -i hallmark_DB_file', '# Read in data\nDE_stats_data <- read.table(template_DE_stats_file, sep="\\t", header=TRUE, row.names=NULL)\n\n# Sort genes by feature 1\n\n# feature 1: numeric vector\n# 5: p-values\n# 6: adjusted p-values\n# 2: logFC\nrank_genes <- as.numeric(as.character(DE_stats_data[,4]))\n\n#print(head(rank_genes))\n\n# feature 2: named vector of gene ids\n# Remove version from gene id\nDE_stats_data[,1] <- gsub("\\\\..*","", DE_stats_data[,1])\n\nnames(rank_genes) <- as.character(DE_stats_data[,1])\n\nprint(head(rank_genes))\n\n## feature 3: decreasing order\nrank_genes = sort(rank_genes, decreasing = TRUE)\n\n#pathway_DB_data <- read.gmt(hallmark_DB_file)\npathway_DB_data <- GSA.read.gmt(hallmark_DB_file)\npathway_parsed <- {}\nfor (i in 1:length(pathway_DB_data$genesets)){\npathway_parsed[pathway_DB_data$geneset.name[i]] <- as.list(pathway_DB_data$genesets[i])\n}\n#print(head(pathway_DB_data))\n# GSEA is a generic gene set enrichment function\n# Different backend methods can be applied depending on the \n# type of annotations\n# Here we will use fgsea\n#enrich_pathways <- GSEA(geneList=rank_genes, \n#                        TERM2GENE=pathway_DB_data,\n#                        nPerm=100000,\n#                        by=\'fgsea\',\n#                        verbose=T)\nenrich_pathways <- fgsea(pathways=pathway_parsed,\n                         stats=rank_genes,\n                         nperm=20000)\nprint(enrich_pathways)\n#plotEnrichment(pathway_parsed[["HALLMARK_P53_PATHWAY"]], stats=rank_genes, gseaParam = 1, ticksSize = 0.2)\n#barplot(sort(rank_genes, decreasing = T))')


# *fgsea* 
# 
# For each of the input pathways, an ES value is calculated. Next, a number
# of random gene sets of the same size are generated, and for each of them
# an ES value is calculated. Then a P-value is estimated as the number of
# random gene sets with the same or more extreme ES value divided by the
# total number of generated gene sets
# 
# What is the ES value that is used?
# Source code for *fgsea* is available [here](https://rdrr.io/bioc/fgsea/src/R/fgsea.R). This package was based on the method published by [Korotkevich et. al.](https://www.biorxiv.org/content/10.1101/060012v2.full.pdf).
# Blog about calulcation is [here](https://www.pathwaycommons.org/guide/primers/data_analysis/gsea/).

# In[15]:


get_ipython().run_cell_magic('R', '-i template_DE_stats_file -i hallmark_DB_file -o template_enriched_pathways', "\nsource('../functions/GSEA_analysis.R')\n\ntemplate_enriched_pathways <- find_enriched_pathways(template_DE_stats_file, hallmark_DB_file)")


# In[16]:


print(template_enriched_pathways.shape)
template_enriched_pathways.head()


# ### Get pathway enrichment for simulated experiments

# In[17]:


# Replace ensembl ids with gene symbols
if rerun_simulated:
    for i in range(num_runs):
        simulated_DE_stats_file = os.path.join(
            local_dir, 
            "DE_stats",
            "DE_stats_simulated_data_"+project_id+"_"+str(i)+".txt")

        utils.replace_ensembl_ids(simulated_DE_stats_file,
                                  gene_id_mapping)


# In[18]:


get_ipython().run_cell_magic('R', '-i project_id -i local_dir -i hallmark_DB_file -i num_runs -i rerun_simulated', '\nsource(\'../functions/GSEA_analysis.R\')\n\nfor (i in 0:(num_runs-1)){\n    simulated_DE_stats_file <- paste(local_dir, \n                                 "DE_stats/DE_stats_simulated_data_", \n                                 project_id,\n                                 "_", \n                                 i,\n                                 ".txt",\n                                 sep="")\n    \n    out_file = paste(local_dir, \n                     "GSEA_stats/GSEA_simulated_data_",\n                     project_id,\n                     "_",\n                     i,\n                     ".txt", \n                     sep="")\n    \n    if (rerun_simulated){\n        enriched_pathways <- find_enriched_pathways(simulated_DE_stats_file, hallmark_DB_file) \n        #print(head(enriched_pathways))\n    \n        write.table(enriched_pathways, file = out_file, row.names = T, sep = "\\t")\n        }\n    }')


# **Check**
# 
# Again, we want to compare our ranked pathways found against what was reported in the original publication.

# ## Get statistics for enriched pathways
# Examine the enriched pathways identified from template experiment -- How are these enriched pathways ranked in the simulated experiments?

# ### Template experiment

# In[19]:


col_to_rank = 'enrichmentScore'


# In[20]:


# Get ranks of template experiment
# Rank pathways by highest enrichment score
template_enriched_pathways['ranking'] = template_enriched_pathways[col_to_rank].rank(ascending = 0) 
template_enriched_pathways = template_enriched_pathways.sort_values(by=col_to_rank, ascending=False)

# Set index to GO ID
template_enriched_pathways.set_index("ID", inplace=True)
print(template_enriched_pathways.shape)
template_enriched_pathways.head()


# In[ ]:


# Check that GO IDs are unique
template_enriched_pathways.index.nunique() == len(template_enriched_pathways)


# ### Simulated experiments

# In[ ]:


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


# In[ ]:


simulated_enriched_pathways_stats = simulated_enriched_pathways_all.groupby(['ID'])[['enrichmentScore', 'pvalue', 'ranking']].agg({
    col_to_rank:['mean', 'std','count'],
    'pvalue':['median'],
    'ranking':['median']
})

simulated_enriched_pathways_stats.head()


# In[ ]:


# Merge template statistics with simulated statistics
template_simulated_enriched_pathways_stats = template_enriched_pathways.merge(simulated_enriched_pathways_stats, 
                                                                              how='outer',
                                                                              left_index=True,
                                                                              right_index=True)
template_simulated_enriched_pathways_stats.head()


# In[ ]:


# Parse columns
median_pval_simulated = template_simulated_enriched_pathways_stats[('pvalue','median')]
median_rank_simulated = template_simulated_enriched_pathways_stats[('ranking','median')]
mean_test_simulated = template_simulated_enriched_pathways_stats[(col_to_rank,'mean')]
std_test_simulated = template_simulated_enriched_pathways_stats[(col_to_rank,'std')]
count_simulated = template_simulated_enriched_pathways_stats[(col_to_rank,'count')]


# ### Calculations for summary table

# In[ ]:


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


# In[ ]:


# Save file
summary_file = os.path.join(
        local_dir, 
        "pathway_summary_table.tsv")

summary.to_csv(summary_file, float_format='%.5f', sep='\t')

