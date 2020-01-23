## Setup environment for recount
## Only run once
install.packages("BiocManager")
install.packages("RMySQL")
BiocManager::install('GenomicFeatures')
BiocManager::install('recount')

## About
# recount2 contains over 70,000 uniformly processed human RNA-seq samples spanning
# TCGA, SRA and GTEx
library('recount')
source("~/Documents/Repos/Batch_effects_simulation/scripts/functions/calculations.R")

## Metadata

# Download the metadata for all samples in recount, and write it to a file
metadata <- all_metadata()
write.table(metadata, '~/Documents/Repos/Batch_effects_simulation/data/metadata/recount2_metadata.tsv', sep='\t', row.names=FALSE)

# Format of the metadata
# Based on definitions from NCBI and blog post: 
# https://www.ncbi.nlm.nih.gov/books/NBK56913/#search.what_do_the_different_sra_accessi
# https://www.ccdatalab.org/blog/2019/3/29/gene-expression-repos-explained
# Project: A sequencing study (i.e. NCBI sequencing study).    
# Sample: Physical biospecimen on which sequencing was performed, biological source material (i.e. HeLa cell line).  
#         A project contains many samples
# Experiment: Unique sequencing library for a specific sample.  A sample can have multiple experiments, most have 1 experiment.
# Run: Sequencing run.  An experiment can contain many runs (i.e. technical replicates)
# In this case, we want to group runs into projects (for "experiment-level" simulation)
project_ids <- unique(metadata$project)

# Entire recount2 is 8TB
# We will only select the top 50 studies instead
selected_project_ids <- sample(project_ids, 50)

## Data

# Get data associated with project ids
# Download the RangedSummarizedExperiment object at the gene level for 
for(i in 1:length(selected_project_ids)){
  if(!file.exists(file.path(selected_project_ids[i], 'rse_gene.Rdata'))) {
    download_study(selected_project_ids[i])
    load(file.path(selected_project_ids[i], 'rse_gene.Rdata'), verbose = TRUE)
  } 
  else {
    load(file.path(selected_project_ids[i], 'rse_gene.Rdata'), verbose = TRUE)
  }
  # Counts are raw read counts from the sequencing run
  # Counts are the number of reads that map that each gene
  
  # Scale counts by total coverage (total reads across the genome) per sample
  # Mixed paired-end and single-end reads per run
  # Most runs are single-end
  # RPKM: normalizes for sequencing depth (coverge per gene) and gene length. Usually used for single-end
  rse_rpkm <- getRPKM(rse_gene, length_var="bp_length")
  rse_tpm <- getTPM(rse_gene, length_var="bp_length")
  
  # Concatenate scaled counts into one matrix
  if (i==1){
    data_counts_all_rpkm <- rse_rpkm
    data_counts_all_tpm <- rse_tpm
  }
  else{
    data_counts_all_rpkm <- cbind(data_counts_all_rpkm, rse_rpkm)
    data_counts_all_tpm <- cbind(data_counts_all_tpm, rse_tpm)
  }
  
  # Rename counts for storage
  assign(paste0("rse_gene_", selected_project_ids[i]), rse_gene)
}

data_counts_all_rpkm <- t(data_counts_all_rpkm)
data_counts_all_tpm <- t(data_counts_all_tpm)

## Save counts matrix to file
  write.table(data_counts_all_rpkm,
            '~/Documents/Repos/Batch_effects_simulation/Human_analysis/data/input/recount2_gene_RPKM_data.tsv',
            sep='\t',
            row.names=TRUE,
            col.names=NA)

#write.table(data_counts_all_tpm,
#            '~/Documents/Repos/Batch_effects_simulation/Human_analysis/data/input/recount2_gene_TPM_data.tsv',
#            sep='\t',
#            row.names=TRUE,
#            col.names=NA)
