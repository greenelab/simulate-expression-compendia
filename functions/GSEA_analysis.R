## Run this once to setup environment
## Used R 3.6.3
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("clusterProfiler")

#library(clusterProfiler)

find_enriched_pathways <- function(DE_stats_file){
    # Read in data
    DE_stats_data <- read.table(DE_stats_file, sep="\t", header=TRUE, row.names=NULL)
   
    # Sort genes by log FC
    
    # feature 1: numeric vector of adjusted p-values
    rank_genes = DE_stats_data[,5]

    # feature 2: named vector of gene ids
    # Remove version from gene id
    DE_stats_data[,1] <- gsub("\\..*","", DE_stats_data[,1])

    names(rank_genes) = as.character(DE_stats_data[,1])

    ## feature 3: decreasing orde
    rank_genes = sort(rank_genes, decreasing = TRUE)
  
    enrich_pathways <- gseGO(
        geneList=rank_genes, 
        ont="ALL", 
        OrgDb=org.Hs.eg.db,
        keyType = "ENSEMBL", 
        verbose=F
        )
  
    return(as.data.frame(enrich_pathways@result))
}

find_over_represented_pathways <- function(DE_stats_file){

  # Read in data
  DE_stats_data <- as.matrix(read.table(DE_stats_file, sep="\t", header=TRUE, row.names=1))
  
  all_genes <- row.names(DE_stats_data)
  print(length((all_genes)))
  
  # Get list of DEGs
  threshold <- 0.05
  sign_DEG_data <- DE_stats_data[DE_stats_data[,'adj.P.Val']<threshold,]
  
  DEG_genes <- row.names(sign_DEG_data)
  print(length((DEG_genes)))
  
  if (length(DEG_genes) > 0){
    # Get over-representations of KEGG pathways in DEG set
    GO_enrich <- enrichGO(
        DEG_genes,
        ont="ALL", OrgDb=org.Hs.eg.db,
        keyType = "ENSEMBL"
        )
    num_GO_pathways <- dim(GO_enrich)[1]
  } else{
    num_GO_pathways <- 0
  }
  print(num_GO_pathways)
  
  return(summary(GO_enrich))
}