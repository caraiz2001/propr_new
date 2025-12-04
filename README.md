# Predicting gene expression in knock-out perturbations using differential proportionality

## Introduction
This project aims to explore how log-ratio based analysis can be used to predict expression counts in unseen perturbations. It is developed inside the framework of the Virtual Cell Challenge (VCC). 

## Dataset

Training data provided by VCC consists of 150 CRISPR knock-out experiments (each with a different targeted gene) resulting in expression matrix (single cell RNA-seq). 

## Motivation
Differential expression (DE) analysis can be used to identify which genes have varying expression levels between conditions (control vs targeted). According to VCC guidelines, Wilcoxon test will be used as the DE ground truth. 

We want to study if it is possible to recapitulate Wilcoxon results using differential proportionality (PROPD), combined across the different perturbations. 

## Implementation

### 1. Running PROPD single on the 150 experiements (control vs target)
PROPD method can be used to identify gene pairs that are at different proportions (i.e, different stoichiometry) bewtween conditions. The pairwise result can be converted into genewise using weighted connectivity. By doing this, we obtain a list of differential expressed genes, equivalent to the one obtained with Wilcoxon test. 



## Project structure

```
/propr_new
│-- data/                  # Files containing gene names (all, targeted)
│-- scripts/               # 
│-- exploratory_analysis/  # Exploratory data analysis scripts & plots
│-- results/               # Results of the 
│   │-- propd_single/  # 
│   │-- propd_pairwise/ #  
|   |-- propd_benchmark/ #
│-- README.rd              # Project documentation
```






