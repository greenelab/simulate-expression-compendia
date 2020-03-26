# Setup 
# Run 1 time
# Using R 3.6

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("clusterProfiler")

library(clusterProfiler)

get_overrep_pathway_table <- function(DE_stats_file,
                              out_file){
  
  # Read in DE summary statistics
  DE_stats_data <- as.matrix(read.table(DE_stats_file, sep="\t", header=TRUE, row.names=1))
  
  all_genes <- row.names(DE_stats_data)
  print(length((all_genes)))
        
  # Get the number of genes that p-value < 0.05/5549
  threshold = 0.05/5549
  sign_DEG_data <- DE_stats_data[DE_stats_data[,'P.Value']<threshold,]
  
  if (dim(sign_DEG_data)[1] > 0){
    
    DEG_genes <- row.names(sign_DEG_data)
    print(length((DEG_genes)))
    
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
    
    head(kegg_enrich)
    print(dim(kegg_enrich))
   
    # Save
    write.table(kegg_enrich, 
                file = out_file, 
                sep = "\t",
                row.names = TRUE,
                col.names = TRUE,
                eol = "\n")
  } else{
    print("No DEGs, cannot perform over-representation analysis")
  }
  
}

#----------------------------------------------------------------------------------------------------------------------------
# Get over-represented pathway table for example experiments generated using original data, experiment-level simulated data,
# sample-level simulated data
main_input_dir="~/UPenn/CGreene/Remote/pathway_analysis/"
DE_stats_original_file <- "~/UPenn/CGreene/Remote/pathway_analysis/output_original/DE_stats_original_data_E-GEOD-51409_example.txt"
DE_stats_simulated_file <- "~/UPenn/CGreene/Remote/pathway_analysis/output_simulated/DE_stats_simulated_data_E-GEOD-51409_example.txt"
DE_stats_control_file <- "~/UPenn/CGreene/Remote/pathway_analysis/output_control/DE_stats_control_data_E-GEOD-51409_example.txt"

out_original_file <- "~/UPenn/CGreene/Remote/pathway_analysis/overrep_pathways_original_data_E-GEOD-51409_example.txt"
out_simulated_file <- "~/UPenn/CGreene/Remote/pathway_analysis/overrep_pathways_simulated_data_E-GEOD-51409_example.txt"
out_control_file <- "~/UPenn/CGreene/Remote/pathway_analysis/overrep_pathways_control_data_E-GEOD-51409_example.txt"

get_overrep_pathway_table(DE_stats_original_file,
                           out_original_file)

get_overrep_pathway_table(DE_stats_simulated_file,
                           out_simulated_file)

get_overrep_pathway_table(DE_stats_control_file,
                           out_control_file)
