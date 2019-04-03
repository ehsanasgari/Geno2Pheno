
# GENO2PHENO PIPELINE

## Summary
Geno2Pheno is the second part of Seq2Geno2Pheno bioinformatics software, which is developped for phenotype prediction and characterization using the genotype information. When the Genotype tables already exist it Geno2Pheno can be used as a stand-alone package.
Geno2Pheno has two main functionalities (i) predictive models of the phenotype (ii) marker detection based on the learned predictive models. The features and functionalities of Geno2Pheno 1.0.0 are detailed in the next sections.

Geno2Pheno works as a data analaysis pipeline that can be customized by the users using a markup format serving as the input of Geno2Pheno, called GeneML. Here are the information will be provided to the Geno2Pheno using GeneML:

![Geno2Pheno](https://user-images.githubusercontent.com/8551117/55457427-243a0080-55ea-11e9-9a55-ce057a2b4b7e.png)

You may see examples of GeneML in the GeneML directory. Seq2Geno automatically generates a GeneML file which can then be modified by the user.

### Running the pipeline

Running the pipeline is pretty straightforward. The only information needed are the GeneML file, number of cores to be used, and if the user wants to override the existing files. Here is an example:

```
python3 geno2pheno.py --genoparse GeneML/sample.genml --override 0 --cores 30
```


### Classification of Phenotype




### Marker detection

Geno2Pheno select top Genotypes based on the overall rankings of selected features in the predictive models and output
a list of important features for further investigations. These markers can be basis for further clinical assays.





