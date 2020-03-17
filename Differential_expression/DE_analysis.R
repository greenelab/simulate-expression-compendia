library("limma")

find_DEGs <- function(metadata_file, experiment_id){

	# Input files
	simulated_filename = "/home/alexandra/Documents/Data/Batch_effects/simulated/analysis_1/selected_simulated_data.txt"
	original_filename = "/home/alexandra/Documents/Data/Batch_effects/simulated/analysis_1/selected_original_data.txt"
	metadata_filename = paste("/home/alexandra/Documents/Data/Batch_effects/simulated/analysis_1/", metadata_file,".txt", sep="")
	

	# Read in data
	simulated_data <- t(as.matrix(read.table(simulated_filename, sep="\t", header=TRUE, row.names=1)))
	original_data <- t(as.matrix(read.table(original_filename, sep="\t", header=TRUE, row.names=1)))
	metadata <- as.matrix(read.table(metadata_filename, sep="\t", header=TRUE, row.names=1))

	group <- interaction(metadata[,1])

	mm <- model.matrix(~0 + group)

	## DEGs of simulated data
	# lmFit fits a linear model using weighted least squares for each gene:
	fit <- lmFit(simulated_data, mm)

	# Comparisons between groups (log fold-changes) are obtained as contrasts of these fitted linear models:
	# Specify which groups to compare:

	if (any(grep(metadata_file,"metadata_deg_phosphate"))){
		contr <- makeContrasts(grouphigh_phosphate - grouplow_phosphate, levels = colnames(coef(fit)))
	}
	else{
		contr <- makeContrasts(group37 - group22, levels = colnames(coef(fit)))
	}
	# Estimate contrast for each gene
	tmp <- contrasts.fit(fit, contr)

	# Empirical Bayes smoothing of standard errors (shrinks standard errors 
	# that are much larger or smaller than those from other genes towards the average standard error)
	tmp <- eBayes(tmp)

	# Get DEGs
	top.table <- topTable(tmp, sort.by = "P", n = Inf)

	# Number of significantly DEGs
	print(length(which(top.table$adj.P.Val < 0.05)))

	# Output signficant DEGs
	out_sim_filename = paste("/home/alexandra/Documents/Data/Batch_effects/simulated/analysis_1/sign_DEG_sim_",experiment_id,".txt",sep="")
	top.table$Gene <- rownames(top.table)
	top.table <- top.table[,c("Gene", names(top.table)[1:6])]
	write.table(top.table, file = out_sim_filename, row.names = F, sep = "\t", quote = F)


	## DEGs of original data
	fit <- lmFit(original_data, mm)
	if (any(grep(metadata_file,"metadata_deg_phosphate"))){
		contr <- makeContrasts(grouphigh_phosphate - grouplow_phosphate, levels = colnames(coef(fit)))
	}
	else{
		contr <- makeContrasts(group37 - group22, levels = colnames(coef(fit)))
	}
	tmp <- contrasts.fit(fit, contr)
	tmp <- eBayes(tmp)

	# Get DEGs
	top.table <- topTable(tmp, sort.by = "P", n = Inf)

	# Number of significantly DEGs
	print(length(which(top.table$adj.P.Val < 0.05)))

	# Output signficant DEGs
	out_original_filename = paste("/home/alexandra/Documents/Data/Batch_effects/simulated/analysis_1/sign_DEG_original_",experiment_id,".txt",sep="")
	top.table$Gene <- rownames(top.table)
	top.table <- top.table[,c("Gene", names(top.table)[1:6])]
	write.table(top.table, file = out_original_filename, row.names = F, sep = "\t", quote = F)
	
}
