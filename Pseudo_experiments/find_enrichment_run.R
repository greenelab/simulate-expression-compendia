# Setup 
# Run 1 time
# Using R 3.6

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("fgsea")

library("fgsea")

find_enriched_pathways <- function(annotation_file,
                                DE_stats_file){
  
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
                    nperm=10000)
  
  # Get the number of genes that adjusted p-value < 0.05
  # These p-values are FDR adjusted
  sign_enrich_data <- fgseaRes[fgseaRes$padj<0.05]
  
  enrich_terms_pathways <-dim(sign_enrich_data)[1]
  print(enrich_terms_pathways)
  
  return(enrich_terms_pathways)
}

#------------------------------------------------------------------------------
# Enriched pathways in example experiment-preserving simulated experiment
pathway_file <- "~/UPenn/CGreene/Remote/pathway_analysis/Pa_KEGG_pathways_ADAGE.txt"
num_enrich_pathways_simulated <- c()
for (i in 0:99){
  DE_stats_simulated_data_file <- paste("~/UPenn/CGreene/Remote/pathway_analysis/output_simulated/DE_stats_simulated_data_E-GEOD-51409_", i, ".txt", sep="")
  cat(paste("running file: ", DE_stats_simulated_data_file, "...\n", sep=""))
  
  run_output <- find_enriched_pathways(pathway_file,
                                       DE_stats_simulated_data_file)
  num_enrich_pathways_simulated <- c(num_enrich_pathways_simulated, run_output)
}


# Enriched pathways in example random-sampling simulated experiment
num_enrich_pathways_control <- c()
for (i in 0:99){
  DE_stats_control_data_file <- paste("~/UPenn/CGreene/Remote/pathway_analysis/output_control/DE_stats_control_data_E-GEOD-51409_", i, ".txt", sep="")
  cat(paste("running file: ", DE_stats_control_data_file, "...\n", sep=""))
  
  run_output <- find_enriched_pathways(pathway_file,
                                       DE_stats_control_data_file)
  
  num_enrich_pathways_control <- c(num_enrich_pathways_control, run_output)
}


# Plot density
library(ggplot2)
df <- data.frame(
  dens = c(num_enrich_pathways_simulated, num_enrich_pathways_control),
  simulation_type = rep(c("experiment-preserving", "random"),each = 100))

f <- ggplot(df, aes(x = dens, fill = simulation_type))+
  geom_density(alpha = 0.3)+
  labs(title="Number of enriched pathways",
       x="Number of enriched pathways",
       y="Count",
       fill="simulation type")
f

# Save 
ggsave("~/UPenn/CGreene/Remote/pathway_analysis/density_enriched_pathways.png", plot = f, dpi=500)
