# Setup 
# Run 1 time
# Using R 3.6

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("clusterProfiler")

library(clusterProfiler)

find_over_representation <- function(DE_stats_file){

  # Read in data
  DE_stats_data <- as.matrix(read.table(DE_stats_file, sep="\t", header=TRUE, row.names=1))
  
  all_genes <- row.names(DE_stats_data)
  print(length((all_genes)))
  
  # Get the number of genes that p-value < 0.05/5549
  adj_threshold = 0.05/5549
  sign_DEG_data <- DE_stats_data[DE_stats_data[,'P.Value']<adj_threshold,]
  
  DEG_genes <- row.names(sign_DEG_data)
  print(length((DEG_genes)))
  
  if (length(DEG_genes) > 0){
    # Get over-representations of KEGG pathways
    kegg_enrich <- enrichKEGG(
      DEG_genes,
      organism = "pae",
      keyType = "kegg",
      pvalueCutoff = 0.05,
      pAdjustMethod = "BH",
      universe = all_genes,
      minGSSize = 10,
      maxGSSize = 500,
      qvalueCutoff = 0.2,
      use_internal_data = FALSE
    )
    num_kegg_pathways <- dim(kegg_enrich)[1]
  } else{
    num_kegg_pathways <- 0
  }
  
  return(num_kegg_pathways)
}

num_over_pathways <- c()

for (i in 0:99){
  DE_stats_simulated_data_file <- paste("~/UPenn/CGreene/Remote/pathway_analysis/output_simulated/DE_stats_simulated_data_E-GEOD-51409_", i, ".txt", sep="")
  cat(paste("running file: ", DE_stats_simulated_data_file, "...\n", sep=""))
  
  run_output <- find_over_representation(DE_stats_simulated_data_file)
  
  num_over_pathways <- c(num_over_pathways, run_output)
}

hist(num_over_pathways)

num_over_pathways <- c()
for (i in 0:99){
  DE_stats_control_data_file <- paste("~/UPenn/CGreene/Remote/pathway_analysis/output_control/DE_stats_control_data_E-GEOD-51409_", i, ".txt", sep="")
  cat(paste("running file: ", DE_stats_control_data_file, "...\n", sep=""))
  
  run_output <- find_over_representation(DE_stats_control_data_file)
  
  num_over_pathways <- c(num_over_pathways, run_output)
}

hist(num_over_pathways)