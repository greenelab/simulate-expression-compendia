import pandas as pd
from scipy import stats

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

    """
    Replaces ensembl gene ids with hgnc symbols 

    Arguments
    ---------
    DE_stats_file: str
        File containing DE stats. Matrix is gene x stats features

    gene_id_mapping: df
        Dataframe mapping ensembl ids (used in DE_stats_file) to hgnc symbols,
        used in Crow et. al.
    """
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

def spearman_ci(gene_rank_df,
num_permutations):
    """
    Returns spearman correlation score and 95% confidence interval

    Arguments
    ---------
    gene_ranking_df: df
        Dataframe containing the our rank and Crow et. al. rank

    num_permutations: int
        The number of permutations to estimate the confidence interval
    """

    r, p = stats.spearmanr(gene_rank_df['Rank (simulated)'],
                gene_rank_df['DE_Prior_Rank'])

    r_perm_values = []
    for i in range(num_permutations):
        
        sample = gene_rank_df.sample(n=len(gene_rank_df), replace=True)
        
        r_perm, p_perm = stats.spearmanr(sample['Rank (simulated)'],
                            sample['DE_Prior_Rank'])
        r_perm_values.append(r_perm)

    sort_r_perm_values = sorted(r_perm_values)
    offset = int(num_permutations*0.025)

    return(r,p,sort_r_perm_values[offset],sort_r_perm_values[num_permutations-offset])