
# coding: utf-8

# # Create compression-only dataset
# This notebook creates an experiment that only applies VAE encoding. This experiment will be used to ...

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
get_ipython().run_line_magic('autoreload', '2')

import os
import glob
import pandas as pd
from sklearn import preprocessing
from keras.models import load_model
from ponyo import utils
from rpy2.robjects import pandas2ri
pandas2ri.activate()


# In[2]:


# Read in config variables
config_file = os.path.abspath(os.path.join(os.getcwd(),"../configs", "config_Pa_experiment_limma.tsv"))
params = utils.read_config(config_file)


# In[3]:


# Load parameters
local_dir = params["local_dir"]
dataset_name = params['dataset_name']
NN_architecture = params['NN_architecture']
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

# Load VAE encoder and decoder models
NN_dir = os.path.join(
    base_dir, 
    dataset_name,
    "models",
    NN_architecture)
model_encoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_encoder_model.h5"))[0]

weights_encoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_encoder_weights.h5"))[0]

model_decoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_decoder_model.h5"))[0]

weights_decoder_file = glob.glob(os.path.join(
    NN_dir,
    "*_decoder_weights.h5"))[0]

loaded_model = load_model(model_encoder_file)
loaded_decode_model = load_model(model_decoder_file)

loaded_model.load_weights(weights_encoder_file)
loaded_decode_model.load_weights(weights_decoder_file)


# In[5]:


# Output files
selected_compressed_data_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "selected_compressed_only_data_"+experiment_id+"_example.txt")

DE_stats_compressed_only_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "output_original",
    "DE_stats_compressed_only_data_"+experiment_id+"_example.txt")


# ## Normalize expression data

# In[6]:


# Read compendium
original_template = pd.read_csv(selected_original_data_file,
                                header=0,
                                index_col=0,
                                sep="\t")

print(original_template.shape)
original_template.head()


# In[7]:


# 0-1 normalize per gene
scaler = preprocessing.MinMaxScaler()
original_data_scaled = scaler.fit_transform(original_template)
original_data_scaled_df = pd.DataFrame(original_data_scaled,
                                columns=original_template.columns,
                                index=original_template.index)

print(original_data_scaled_df.shape)
original_data_scaled_df.head()


# ## Encode and decode data

# In[8]:


# Pass original data through VAE
# Encode selected experiment into latent space
data_encoded = loaded_model.predict_on_batch(original_data_scaled_df)
data_encoded_df = pd.DataFrame(
    data_encoded, 
    index=original_data_scaled_df.index)

# Decode simulated data into raw gene space
data_decoded = loaded_decode_model.predict_on_batch(data_encoded_df)

vae_data = pd.DataFrame(data_decoded,
                        index=data_encoded_df.index,
                        columns=original_data_scaled_df.columns)

print(vae_data.shape)
vae_data.head()


# In[9]:


# Scale data back into original range for DE analysis
vae_data_scaled = scaler.inverse_transform(vae_data)

vae_data_scaled_df = pd.DataFrame(
    vae_data_scaled,
    columns=vae_data.columns,
    index=vae_data.index
)


# In[10]:


vae_data_scaled_df.head()


# In[11]:


# Save expression data for use in heatmap plot
vae_data_scaled_df.to_csv(selected_compressed_data_file, sep="\t")


# ## DE analysis

# In[12]:


get_ipython().run_cell_magic('R', '', '#if (!requireNamespace("BiocManager", quietly = TRUE))\n#  install.packages("BiocManager")\n\n#BiocManager::install("limma")')


# In[13]:


get_ipython().run_cell_magic('R', '', 'suppressPackageStartupMessages(library("limma"))')


# In[14]:


# files for analysis
metadata_file = os.path.join(
    local_dir,
    "pseudo_experiment",
    "metadata_deg_temp.txt")


# In[15]:


get_ipython().run_cell_magic('R', '-i metadata_file -i experiment_id -i selected_compressed_data_file -i DE_stats_compressed_only_file', 'get_DE_stats <- function(metadata_file, \n                         experiment_id, \n                         expression_file,\n                         out_file){\n    # Read in data\n    expression_data <- t(as.matrix(read.csv(expression_file, sep="\\t", header=TRUE, row.names=1)))\n    metadata <- as.matrix(read.csv(metadata_file, sep="\\t", header=TRUE, row.names=1))\n    \n    print("Checking sample ordering...")\n    print(all.equal(colnames(expression_data), rownames(metadata)))\n  \n    # NOTE: It make sure the metadata is in the same order \n    # as the column names of the expression matrix.\n    group <- interaction(metadata[,1])\n  \n    mm <- model.matrix(~0 + group)\n  \n    ## DEGs of simulated data\n    # lmFit expects input array to have structure: gene x sample\n    # lmFit fits a linear model using weighted least squares for each gene:\n    fit <- lmFit(expression_data, mm)\n  \n    # Comparisons between groups (log fold-changes) are obtained as contrasts of these fitted linear models:\n    # Samples are grouped based on experimental condition\n    # The variability of gene expression is compared between these groups\n    # For experiment E-GEOD-51409, we are comparing the expression profile\n    # of samples grown in 37 degrees versus those grown in 22 degrees\n    contr <- makeContrasts(group37 - group22, levels = colnames(coef(fit)))\n\n    # Estimate contrast for each gene\n    tmp <- contrasts.fit(fit, contr)\n\n    # Empirical Bayes smoothing of standard errors (shrinks standard errors \n    # that are much larger or smaller than those from other genes towards the average standard error)\n    tmp <- eBayes(tmp)\n  \n    # Get significant DEGs\n    top.table <- topTable(tmp, sort.by = "P", n = Inf)\n    all_genes <-  as.data.frame(top.table)\n  \n    # Find all DEGs based on Bonferroni corrected p-value cutoff\n    threshold = 0.05/5549\n    num_sign_DEGs <- all_genes[all_genes[,\'P.Value\']<threshold,]\n  \n  # Save summary statistics of DEGs\n  write.table(all_genes, file = out_file, row.names = T, sep = "\\t", quote = F)\n  \n}\n\nget_DE_stats(metadata_file, experiment_id, selected_compressed_data_file, DE_stats_compressed_only_file)')

