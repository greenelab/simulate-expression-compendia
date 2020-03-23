# Setup 
# Run 1 time
# Using R 3.6

#if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

#BiocManager::install("fgsea")

library("fgsea")

get_enriched_pathway_table <- function(annotation_file,
                                       DE_stats_file,
                                       out_file){
  
  # Read DE summary statistics file
  DE_stats_data <- as.matrix(read.table(DE_stats_file, sep="\t", header=TRUE, row.names=1))
  
  # Read annotation file
  pathway <- read.table(annotation_file, sep = "\t", header = F, row.names = 1,
                        stringsAsFactors = F)
  pathway_names <- as.list(rownames(pathway))
  
  # Convert pathway data into proper format, which looks to be a dictionary
  # Pathway name : list of genes in the pathway
  # Note: parsing is currently based on the format obtained from annotation files 
  # downloaded from ADAGE
  pathway_parsed <- {}
  for (i in 1:nrow(pathway)){
    pathway_parsed[i] <-pathway_names[i]
  }
  for (i in 1:nrow(pathway)){
    pathway_parsed[paste(pathway_names[i])] <- strsplit(pathway[i,2],";")
  }
  
  # Rank genes by logFC
  gene_ranks <- DE_stats_data[,'logFC']
  
  # Pathways: dictionary {pathway name: list of genes}
  # stats: array with gene_id - numeric score
  # minsize, maxsize: size of a gene set to test
  fgseaRes <- fgsea(pathways = pathway_parsed, 
                    stats = gene_ranks,
                    minSize=15,
                    maxSize=500,
                    nperm=5000)
  
  # Sort results by adjusted p-value
  sorted_fgseaRes <- fgseaRes[order(fgseaRes$padj),]
  
  # Get top 10 enriched pathways
  top_pathways <- sorted_fgseaRes[0:10]
  top_pathways = subset(top_pathways, select = -c(leadingEdge))
  #top_pathways$leadingEdge <- unlist(top_pathways$leadingEdge, use.names = FALSE) 
  print(top_pathways)
  
  # Save
  write.table(top_pathways, 
              file = out_file, 
              sep = "\t",
              row.names = TRUE,
              col.names = TRUE,
              eol = "\n")
}

#------------------------------------------------------------------------------
# Enriched pathways in example experiment-preserving simulated experiment
annotation_file <- "~/UPenn/CGreene/Remote/pathway_analysis/Pa_KEGG_pathways_ADAGE.txt"
DE_stats_original_file <- "~/UPenn/CGreene/Remote/pathway_analysis/output_original/DE_stats_original_data_E-GEOD-51409_example.txt"
DE_stats_simulated_file <- "~/UPenn/CGreene/Remote/pathway_analysis/output_simulated/DE_stats_simulated_data_E-GEOD-51409_example.txt"
DE_stats_control_file <- "~/UPenn/CGreene/Remote/pathway_analysis/output_control/DE_stats_control_data_E-GEOD-51409_example.txt"

out_original_file <- "~/UPenn/CGreene/Remote/pathway_analysis/output_original/enriched_pathways_original_data_E-GEOD-51409_example.txt"
out_simulated_file <- "~/UPenn/CGreene/Remote/pathway_analysis/output_simulated/enriched_pathways_simulated_data_E-GEOD-51409_example.txt"
out_control_file <- "~/UPenn/CGreene/Remote/pathway_analysis/output_control/enriched_pathways_control_data_E-GEOD-51409_example.txt"

get_enriched_pathway_table(annotation_file,
                           DE_stats_original_file,
                           out_original_file)

get_enriched_pathway_table(annotation_file,
                           DE_stats_simulated_file,
                           out_simulated_file)

get_enriched_pathway_table(annotation_file,
                           DE_stats_control_file,
                           out_control_file)
