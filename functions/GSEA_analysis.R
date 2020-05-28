## Run this once to setup environment
## Used R 3.6.3
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("clusterProfiler")

#library(clusterProfiler)

convertEnsembl2Symbol <- function(ensembl.genes) {
  #require(biomaRt)
  ensembl = useMart("ensembl",dataset="hsapiens_gene_ensembl")
  getBM(values = ensembl.genes, attributes = c('ensembl_gene_id','hgnc_symbol'), 
        filters = 'ensembl_gene_id', mart = ensembl, bmHeader=FALSE )
}

convertEnsembl2Entrez <- function(ensembl.genes) {
  #require(biomaRt)
  ensembl = useMart("ensembl",dataset="hsapiens_gene_ensembl")
  getBM(values = ensembl.genes, attributes = c('ensembl_gene_id','entrezgene_id'), 
        filters = 'ensembl_gene_id', mart = ensembl, bmHeader=FALSE )
}

get_ensembl_symbol_mapping <- function(DE_stats_file){
    # Read in data
    DE_stats_data <- read.table(DE_stats_file, sep="\t", header=TRUE, row.names=NULL)
    
    # Remove version from gene id
    DE_stats_data[,1] <- gsub("\\..*","", DE_stats_data[,1])
    
    # Get mapping from ensembl - gene symbol
    gene_id_mapping <- convertEnsembl2Symbol(as.character(DE_stats_data[,1]))
    
    return(gene_id_mapping)
    }

find_enriched_pathways <- function(DE_stats_file,
                                  pathway_DB){
    # Read in data
    DE_stats_data <- read.table(DE_stats_file, sep="\t", header=TRUE, row.names=NULL)
   
    # Sort genes by feature 1
    
    # feature 1: numeric vector
    # 5: p-values
    # 6: adjusted p-values
    # 2: logFC
    rank_genes <- as.numeric(as.character(DE_stats_data[,2]))
    
    #print(head(rank_genes))

    # feature 2: named vector of gene ids
    # Remove version from gene id
    DE_stats_data[,1] <- gsub("\\..*","", DE_stats_data[,1])

    names(rank_genes) <- as.character(DE_stats_data[,1])

    #print(head(rank_genes))

    ## feature 3: decreasing order
    #rank_genes = sort(rank_genes, decreasing = TRUE)

    #barplot(sort(rank_genes, decreasing = T))
  
    #enrich_pathways <- gseGO(
    #    geneList=rank_genes, 
    #    ont="ALL", 
    #    OrgDb=org.Hs.eg.db,
    #    keyType = "ENSEMBL", 
    #    verbose=F
    #    )
    #pathway_DB_data <- read.gmt(pathway_DB)
    pathway_DB_data <- GSA.read.gmt(hallmark_DB_file)
    pathway_parsed <- {}
    for (i in 1:length(pathway_DB_data$genesets)){
    pathway_parsed[pathway_DB_data$geneset.name[i]] <- as.list(pathway_DB_data$genesets[i])
    }

    #print(head(pathway_DB_data))
    #enrich_pathways <- GSEA(geneList=rank_genes, 
    #                        TERM2GENE=pathway_DB_data,
    #                        nPerm=1000,
    #                        verbose=T)
    enrich_pathways <- fgsea(pathways=pathway_parsed,
                              stats=rank_genes,
                              nperm=1000)

    #plotEnrichment(pathway_parsed[["HALLMARK_ADIPOGENESIS"]], stats=rank_genes, gseaParam = 1, ticksSize = 0.2)

    return(as.data.frame(enrich_pathways))
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