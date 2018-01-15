
# Antibiotic Resistance Phenotype  <br> prediction and analysis <br> for Pseudomonas Aeruginosa

Ehsaneddin Asgari
ehaneddin.asgari@helmholtz-hzi.de
<hr>
In this package you can find the following modules:

## PhyloChi<sup>2<sub>
The aim of this program is to find the relevant genotype gains and losses to the resistance phenotype over the edges of phylogenetic tree using chi-square statistics with fdr correction. To calculate the relevant features faster, you may specify number of cores and it can run in parallel (using 10 cores runs in ~10 min). Example:

    PhyChi=PhyloChi2('data_config/tree.nwk')
    PhyChi.generate_parallel_gainloss_data_for_drug(drug_idx, num_p=20)
The results will be generated in the result path (PhyloChi2.saving_path).
This module uses the data_access/creation pipeline of the project described later.
Find the extracted features:

> results/feature_selection/phylochi2/

Other methods implemented here for important features are the following:

 1. Chi<sup>2</sup> selected features (without phylogenetic tree), you can find the extracted features using different settings (for single drugs and multiple drugs) at:

> results/feature_selection/chi2/

 3. Random Forest features:
> results/feature_selection/random_forest/

![screen shot 2018-01-16 at 12 27 15 am](https://user-images.githubusercontent.com/8551117/34965448-656175f4-fa54-11e7-8e62-75d7942086e0.png)



## Classifier
Provides a framework based on sklearn for tuning and evaluation of classical classification algorithms including:
 - Random Forest
 - SVM
 - KNN
 - Deep Neural Network

Classification/regression schemes:
<li>Single drugs
<li>Multiple drugs ⇒ multiclass-multilabel classifications
<li>MIC value regression

Stratified k-fold as well as nested cross-validations have been explored
This module uses the data_access/creation pipeline of the project described later.

## Feature clustering
Hierarchical clustering of important features (based on their co-occurances) to find their causal relations as well as selecting the representative SNPs.

![screen shot 2018-01-16 at 12 33 57 am](https://user-images.githubusercontent.com/8551117/34965551-fd190be6-fa54-11e7-9a50-46a89f2a6ff5.png)



## Phylogenetic representation as input
Based on the the phylogenetic distance using multidimensional scaling project isolates to 100D space and use it as a new feature.

![screen shot 2018-01-16 at 12 31 13 am](https://user-images.githubusercontent.com/8551117/34965482-9a1915e0-fa54-11e7-862f-d0ce67495ff8.png)


## Drug analysis

 1. Drug relations using information theoretic measures
 2. Drug relations using rule association mining
 
 ![screen shot 2018-01-16 at 12 32 24 am](https://user-images.githubusercontent.com/8551117/34965514-c2aa0f3c-fa54-11e7-9aac-ce8b56e4cf75.png)


## Isolates analysis

 1. Isolates clustering using information theoretic measures
 

## Pipeline for data creation and accession




