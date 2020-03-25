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
  # lmFit expects input array to have structure: gene x sample
  # lmFit fits a linear model using weighted least squares for each gene:
  fit <- lmFit(expression_data, mm)
  
  # Comparisons between groups (log fold-changes) are obtained as contrasts of these fitted linear models:
  # Samples are grouped based on experimental condition
  # The variability of gene expression is compared between these groups
  if (grep("metadata_deg_temp", metadata_file)){
    # For experiment E-GEOD-51409, we are comparing the expression profile
    # of samples grown in 37 degrees versus those grown in 22 degrees
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
  main_out_dir="~/UPenn/CGreene/Remote/pathway_analysis/"
  if (data_type == "control"){
    #out_sim_filename = paste("/home/alexandra/Documents/Data/Batch_effects/pseudo_experiment/output_control/DEG_control_data_E-GEOD-51409_", run, ".txt", sep="")
    out_sim_filename = paste(main_out_dir, "output_control/DE_stats_control_data_E-GEOD-51409_", run, ".txt", sep="")
  } else if (data_type == "simulated"){
    #out_sim_filename = paste("/home/alexandra/Documents/Data/Batch_effects/pseudo_experiment/output_simulated/DEG_simulated_data_E-GEOD-51409_", run, ".txt", sep="")
    out_sim_filename = paste(main_out_dir, "output_simulated/DE_stats_simulated_data_E-GEOD-51409_", run, ".txt", sep="")
  } else {
    #out_sim_filename = paste("/home/alexandra/Documents/Data/Batch_effects/pseudo_experiment/output_original/DEG_original_data_E-GEOD-51409_", run, ".txt", sep="")
    out_sim_filename = paste(main_out_dir, "output_original/DE_stats_original_data_E-GEOD-51409_", run, ".txt", sep="")
    
  }  
  write.table(all_genes, file = out_sim_filename, row.names = T, sep = "\t", quote = F)
  
  return(nrow(sign_DEGs))
  
}

#-------------------------------------------------------------------------
# Get DE stats for representative example to generate heatmap
main_input_dir="~/UPenn/CGreene/Remote/DE_analysis/"
metadata_file <- paste(main_input_dir, "metadata_deg_temp.txt", sep="")
selected_simulated_data_file <- paste(main_input_dir, "selected_control/selected_control_data_E-GEOD-51409_example.txt", sep="")
selected_simulated_data_file <- paste(main_input_dir, "selected_simulated/selected_simulated_data_E-GEOD-51409_example.txt", sep="")
selected_original_data_file <- paste(main_input_dir, "selected_original/selected_original_data_E-GEOD-51409_example.txt", sep="")
experiment_id <- "E-GEOD-51409"

cat(paste("running file: ", selected_control_data_file, "...\n", sep=""))
run_output <- get_DE_stats(metadata_file,
                           experiment_id, 
                           selected_control_data_file,
                           "control",
                           "example")
cat(run_output)

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

# Create boxplot for the number of DEGs (based on adj p-value<0.05 only)
library(ggplot2)

name_control <- rep("sample-lvl", 100)
name_sim <- rep("experiment-lvl", 100)
names <- append(name_control, name_sim)
num_DEGs <- append(num_sign_DEGs_control, num_sign_DEGs_sim)

df <- data.frame(num_DEGs, names)

p <- ggplot(df, aes(x=names, y=num_DEGs, color=names)) + 
  geom_boxplot() +
  labs(title="Differential expression across multiple simulated experiments",
       x="Simulation type",
       y = "Number of differentially expressed genes",
       color = "simulation type")+
  scale_color_manual(values=c("#E69F00", "#56B4E9"))
p

# Save 
ggsave(main_input_dir,"boxplot_num_DEGs.png", plot = p, dpi=500)

#------------------------------------------------------------------
# Get DE statistics for multiple simulated and control datasets
num_sign_DEGs_control <- c()
for (i in 0:99){
  metadata_file <- paste(main_input_dir, "metadata_deg_temp.txt", sep="")
  selected_control_data_file <- paste(main_input_dir, "selected_control/selected_control_data_E-GEOD-51409_", i, ".txt", sep="")
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
for (i in 0:99){
  metadata_file <- paste(main_input_dir, "metadata_deg_temp.txt", sep="")
  selected_simulated_data_file <- paste(main_input_dir, "selected_simulated/selected_simulated_data_E-GEOD-51409_", i, ".txt", sep="")
  experiment_id <- "E-GEOD-51409"
  cat(paste("running file: ", selected_simulated_data_file, "...\n", sep=""))
  
  run_output <- get_DE_stats(metadata_file,
                             experiment_id, 
                             selected_simulated_data_file,
                             "simulated",
                             i)
  
  num_sign_DEGs_sim <- c(num_sign_DEGs_sim, run_output)
}
median(num_sign_DEGs_sim)
