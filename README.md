# The impact of undesired technical variability on large-scale data compendia

**Alexandra Lee and Casey Greene 2020**

**University of Pennsylvania**

This repository stores data and analysis modules to simulate compendia of gene expression data and measure the effect of technical sources of variation on our ability to extract an underlying biological signal.  

*Motivation:* In the last two decades, scientists working in different labs have assayed gene expression from millions of samples. These experiments can be combined into a compendium and used to extract novel biological patterns. However, combining different experiments introduces technical variance, which could distort biological patterns and lead to misinterpretation. As the scale and prevalence of these compendia increases, it becomes crucial to evaluate how integrating multiple experiments affects our ability to detect biological patterns.

*Objective:* To determine the extent to which underlying biological structures are masked by technical variants via simulation of a multi-experiment compendia.

*Method:* We used a generative multi-layer neural network to simulate a compendium of P. aeruginosa gene expression experiments. We performed a pairwise comparison of the simulated compendium versus the simulated compendium containing varying number of sources of technical variation.

*Results:* We found that it was difficult to detect the original biological structure of interest in a compendium containing some sources of technical variation unless we applied batch correction. Interestingly, as the number of sources of variation increased, it became easier to detect the original biological structure without correction. Furthermore, when we applied batch correction, it reduced our power to detect the biological structure of interest.     

*Conclusion:* When combining some sources of technical variation, it is best to perform batch correction. However, as the number of sources increases, batch correction becomes unnecessary and indeed harms our ability to extract biological patterns.

## Analysis Modules

There are 2 analyses using Pseudomonas dataset in the `Pseudomonas_analysis` directory and 2 analyses using the recount2 dataset in the `Human_analysis` directory:

| Name | Description |
| :--- | :---------- |
| [analysis_0](Pseudomonas_analysis/analysis_0/) | Pseudomonas sample-level gene expression simulation|
| [analysis_1](Pseudomonas_analysis/analysis_1/) | Pseudomonas experiment-level gene expression simulation|
| [analysis_2](Human_analysis/analysis_2/) | Human sample-level gene expression simulation|
| [analysis_3](Human_analysis/analysis_3/) | Human experiment-level gene expression simulation|


## Computational Environment

All processing and analysis scripts were performed using the conda environment specified in `environment.yml`.
To build and activate this environment run:

```bash
# conda version 4.6.12
conda env create -f environment.yml

conda activate batch_effects
```

## Acknowledgements
We would like to thank YoSon Park and David Nicholson for insightful discussions and code review