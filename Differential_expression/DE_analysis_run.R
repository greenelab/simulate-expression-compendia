# Setup 
# Run 1 time
# Using R 3.6

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("limma")

library("limma")

get_DE_stats <- function(metadata_file, 
                      experiment_id, 
                      expression_file,
                      data_type,
                      run){
  
  # Read in data
  expression_data <- t(as.matrix(read.table(expression_file, sep="\t", header=TRUE, row.names=1)))
  metadata <- as.matrix(read.table(metadata_file, sep="\t", header=TRUE, row.names=1))
  
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
  # These p-values are FDR adjusted
  sign_DEGs <- all_genes[all_genes[,'adj.P.Val']<0.05,]
  
  # Save summary statistics of DEGs
  if (data_type == "control"){
    #out_sim_filename = paste("/home/alexandra/Documents/Data/Batch_effects/pseudo_experiment/output_control/DEG_control_data_E-GEOD-51409_", run, ".txt", sep="")
    out_sim_filename = paste("~/UPenn/CGreene/Remote/pathway_analysis/output_control/DE_stats_control_data_E-GEOD-51409_", run, ".txt", sep="")
  } else if (data_type == "simulated"){
    #out_sim_filename = paste("/home/alexandra/Documents/Data/Batch_effects/pseudo_experiment/output_simulated/DEG_simulated_data_E-GEOD-51409_", run, ".txt", sep="")
    out_sim_filename = paste("~/UPenn/CGreene/Remote/pathway_analysis/output_simulated/DE_stats_simulated_data_E-GEOD-51409_", run, ".txt", sep="")
  } else {
    #out_sim_filename = paste("/home/alexandra/Documents/Data/Batch_effects/pseudo_experiment/output_original/DEG_original_data_E-GEOD-51409_", run, ".txt", sep="")
    out_sim_filename = paste("~/UPenn/CGreene/Remote/pathway_analysis/output_original/DE_stats_original_data_E-GEOD-51409_", run, ".txt", sep="")
    
  }  
  write.table(sign_DEGs, file = out_sim_filename, row.names = T, sep = "\t", quote = F)
  
  return(nrow(sign_DEGs))
  
}

#-------------------------------------------------
# Get DE stats for heatmap
## Paths based on laptop directory
metadata_file <- "~/UPenn/CGreene/Remote/DE_analysis/metadata_deg_temp.txt"
selected_simulated_data_file <- "~/UPenn/CGreene/Remote/DE_analysis/selected_simulated/selected_simulated_data_E-GEOD-51409_example.txt"
selected_original_data_file <- "~/UPenn/CGreene/Remote/DE_analysis/selected_original/selected_original_data_E-GEOD-51409_example.txt"
experiment_id <- "E-GEOD-51409"
cat(paste("running file: ", selected_simulated_data_file, "...\n", sep=""))
  
run_output <- get_DE_stats(metadata_file,
                           experiment_id, 
                           selected_simulated_data_file,
                           "simulated",
                           "example")
cat(run_output)
cat(paste("running file: ", selected_original_data_file, "...\n", sep=""))

run_output <- get_DE_stats(metadata_file,
                           experiment_id, 
                           selected_original_data_file,
                           "original",
                           "example")
cat(run_output)

#---------------------------------------------------

# Get DE statistics for multiple simulated and control datasets
## Paths based on laptop directory
num_sign_DEGs <- c()
for (i in 0:99){
  metadata_file <- "~/UPenn/CGreene/Remote/DE_analysis/metadata_deg_temp.txt"
  selected_control_data_file <- paste("~/UPenn/CGreene/Remote/DE_analysis/selected_control/selected_control_data_E-GEOD-51409_", i, ".txt", sep="")
  experiment_id <- "E-GEOD-51409"
  cat(paste("running file: ", selected_control_data_file, "...\n", sep=""))
  
  run_output <- get_DE_stats(metadata_file,
                              experiment_id, 
                              selected_control_data_file,
                              "control",
                              i)
  
  num_sign_DEGs_control <- c(num_sign_DEGs_control, run_output)
}
median(num_sign_DEGs_control)
sum(num_sign_DEGs_control)

num_sign_DEGs_sim <- c()
## Paths based on laptop directory
for (i in 0:99){
  metadata_file <- "~/UPenn/CGreene/Remote/DE_analysis/metadata_deg_temp.txt"
  selected_simulated_data_file <- paste("~/UPenn/CGreene/Remote/DE_analysis/selected_simulated/selected_simulated_data_E-GEOD-51409_", i, ".txt", sep="")
  experiment_id <- "E-GEOD-51409"
  cat(paste("running file: ", selected_simulated_data_file, "...\n", sep=""))
  
  run_output <- get_DE_stats(metadata_file,
                             experiment_id, 
                             selected_simulated_data_file,
                             "control",
                             i)
  
  num_sign_DEGs_sim <- c(num_sign_DEGs_sim, run_output)
}
median(num_sign_DEGs_sim)
