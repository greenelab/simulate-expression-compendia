# Setup 
# Run 1 time
# Using R 3.6

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("limma")

library("limma")

find_DEGs_run <- function(metadata_file, 
                      experiment_id, 
                      expression_file,
                      data_type,
                      run){
  
  # Input files
  expression_filename = expression_file
  metadata_filename = metadata_file
  
  
  # Read in data
  expression_data <- t(as.matrix(read.table(expression_filename, sep="\t", header=TRUE, row.names=1)))
  metadata <- as.matrix(read.table(metadata_filename, sep="\t", header=TRUE, row.names=1))
  
  # NOTE: It is very,very,very important that you make sure the metadata is in the same order 
  # as the column names of the expression matrix.
  group <- interaction(metadata[,1])
  
  mm <- model.matrix(~0 + group)
  
  ## DEGs of simulated data
  # lmFit fits a linear model using weighted least squares for each gene:
  fit <- lmFit(expression_data, mm)
  
  # Comparisons between groups (log fold-changes) are obtained as contrasts of these fitted linear models:
  # Specify which groups to compare:
  
  if (grep("metadata_deg_temp", metadata_file)){
    contr <- makeContrasts(group37 - group22, levels = colnames(coef(fit)))
  }
  # Estimate contrast for each gene
  tmp <- contrasts.fit(fit, contr)
  
  # Empirical Bayes smoothing of standard errors (shrinks standard errors 
  # that are much larger or smaller than those from other genes towards the average standard error)
  tmp <- eBayes(tmp)
  
  # Get significant DEGs
  top.table <- topTable(tmp, sort.by = "P", n = Inf)
  all_genes <-  as.data.frame(top.table)
  
  # Find all DEGs based on adjusted p-value cutoff
  adj_threshold = 0.05/5549
  sign_DEGs <- all_genes[all_genes[,'P.Value']<adj_threshold,]
  
  # Save summary statistics of DEGs
  if (data_type == "control"){
    out_sim_filename = paste("/home/alexandra/Documents/Data/Batch_effects/pseudo_experiment/output_control/DE_stats_control_data_E-GEOD-51409_", run, ".txt", sep="")
  } else if (data_type == "simulated"){
    out_sim_filename = paste("/home/alexandra/Documents/Data/Batch_effects/pseudo_experiment/output_simulated/DE_stats_simulated_data_E-GEOD-51409_", run, ".txt", sep="")
  } else{
    out_sim_filename = paste("/home/alexandra/Documents/Data/Batch_effects/pseudo_experiment/output_original/DE_stats_original_data_E-GEOD-51409_", run, ".txt", sep="")
  }  
  write.table(all_genes, file = out_sim_filename, row.names = T, sep = "\t", quote = F)
  
  return(nrow(sign_DEGs))
  
}