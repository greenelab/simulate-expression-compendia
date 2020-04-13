# Setup 
# Run 1 time
# Using R 3.6

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("clusterProfiler")

library(clusterProfiler)

find_over_represented_pathways <- function(DE_stats_file){

  # Read in data
  DE_stats_data <- as.matrix(read.table(DE_stats_file, sep="\t", header=TRUE, row.names=1))
  
  all_genes <- row.names(DE_stats_data)
  print(length((all_genes)))
  
  # Get list of DEGs
  # DEGs are those genes where p-value < 0.05/5549 (Bonferroni multi-test correction)
  threshold <- 0.05/5549
  sign_DEG_data <- DE_stats_data[DE_stats_data[,'P.Value']<threshold,]
  
  DEG_genes <- row.names(sign_DEG_data)
  print(length((DEG_genes)))
  
  if (length(DEG_genes) > 0){
    # Get over-representations of KEGG pathways in DEG set
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
  print(num_kegg_pathways)
  
  return(num_kegg_pathways)
}

#------------------------------------------------------------------------------------------------------------------------------------
# Calculate over-representation of KEGG pathways for multiple simulated experiments using experiment-level 
# approach ("simulated") and using sample-level approach ("control")
main_input_dir="~/UPenn/CGreene/Remote/pathway_analysis/"

num_over_pathways_simulated <- c()
for (i in 0:99){
  DE_stats_simulated_data_file <- paste(main_input_dir, "output_simulated/DE_stats_simulated_data_E-GEOD-51409_", i, ".txt", sep="")
  cat(paste("running file: ", DE_stats_simulated_data_file, "...\n", sep=""))
  
  run_output <- find_over_represented_pathways(DE_stats_simulated_data_file)
  
  num_over_pathways_simulated <- c(num_over_pathways_simulated, run_output)
}


num_over_pathways_control <- c()
for (i in 0:99){
  DE_stats_control_data_file <- paste(main_input_dir, "output_control/DE_stats_control_data_E-GEOD-51409_", i, ".txt", sep="")
  cat(paste("running file: ", DE_stats_control_data_file, "...\n", sep=""))
  
  run_output <- find_over_represented_pathways(DE_stats_control_data_file)
  
  num_over_pathways_control <- c(num_over_pathways_control, run_output)
}

# Create boxplot for the number of DEGs (based on adj p-value<0.05 only)
install.packages("svglite")
library(svglite)
library(ggplot2)

name_control <- rep("sample-lvl", 100)
name_sim <- rep("experiment-lvl", 100)
names <- append(name_control, name_sim)
num_over_pathways <- append(num_over_pathways_control, num_over_pathways_simulated)

df <- data.frame(num_over_pathways, names)

p <- ggplot(df, aes(x=names, y=num_over_pathways, color=names)) + 
  geom_boxplot() +
  labs(title="Number of over-represented pathways across multiple simulated experiments",
       x="Simulation type",
       y = "Number of over-represented pathways",
       color = "simulation type")+
  scale_color_manual(values=c("#E69F00", "#56B4E9")) +
  theme(
    legend.title=element_text(family='sans-serif', size=15),
    legend.text=element_text(family='sans-serif', size=12),
    plot.title=element_text(family='sans-serif', size=15),
    axis.text=element_text(family='sans-serif', size=12),
    axis.title=element_text(family='sans-serif', size=15)
  )
p


# Save 
ggsave(paste(main_input_dir,"boxplot_num_over_represented_pathways.svg", sep=""), plot = p, device="svg")