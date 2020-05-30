import pandas as pd

def read_config(filename):
    """
    Read and parse configuration file containing stored user variables

    These variables are then passed to the analysis notebooks
    """
    f = open(filename)
    config_dict = {}
    for lines in f:
        items = lines.split('\t', 1)
        config_dict[items[0]] = eval(items[1])
    return config_dict

def replace_ensembl_ids(DE_stats_file,
gene_id_mapping):
    # Read in data
    DE_stats = pd.read_csv(
        DE_stats_file,
        header=0,
        sep='\t',
        index_col=0)

    # There is the same ensembl id that maps to different gene symbols

    # Manually checked a few duplicates and found that
    # 1. Some ensembl ids are mapped to same gene symbol twice
    # 2. Some duplicates are due to different version numbers.
    # Higher ensembl version number (i.e. more updated) corresponds to the
    # first gene id symbol. So we can remove the second occurring duplicate

    # Doesn't appear that there exists a conversion using version numbers
    
    #gene_id_mapping[gene_id_mapping.index.duplicated()]
    #gene_id_mapping[gene_id_mapping.index == 'ENSG00000276085']
    #gene_id_mapping[gene_id_mapping.index == 'ENSG00000230417']
    #gene_id_mapping[gene_id_mapping.index == 'ENSG00000124334']
    #gene_id_mapping[gene_id_mapping.index == 'ENSG00000223484']

    # Lookup version of duplicated ensembls 
    #template_DE_stats[template_DE_stats.index.str.contains("ENSG00000276085")]

    # Keep first occurence of duplicated ensembl ids
    gene_id_mapping = gene_id_mapping.loc[~gene_id_mapping.index.duplicated(keep='first')]

    # Format ensembl ids
    # Remove version number 
    DE_stats.index = DE_stats.index.str.split(".").str[0]

    # Replace ensembl ids with gene symbol
    DE_stats.index = DE_stats.index.map(gene_id_mapping['hgnc_symbol'])

    # Remove rows where we couldn't map ensembl id to gene symbol
    DE_stats = DE_stats[~(DE_stats.index == "")]

    # Save
    DE_stats.to_csv(DE_stats_file, float_format='%.5f', sep='\t')

def get_gene_id_mapping(genes_to_map,
gene_id_dict_file):
    gene_ids_hgnc = {}
    for gene_id in gene_ids:
        gene_id_strip = gene_id.split(".")[0]
        if gene_id_strip in list(gene_id_mapping.index):
            if len(gene_id_mapping.loc[gene_id_strip]) > 1:
                gene_ids_hgnc[gene_id] = gene_id_mapping.loc[gene_id_strip].iloc[0][0]
            else:
                gene_ids_hgnc[gene_id] = gene_id_mapping.loc[gene_id_strip][0]

    gene_ids_hgnc
    outfile = open(gene_id_dict_file,'wb')
    pickle.dump(gene_ids_hgnc,outfile)
    outfile.close()