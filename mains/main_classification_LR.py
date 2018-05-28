__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__project__ = "GENO2PHENO of SEQ2GENO2PHENO"
__website__ = ""

import sys

sys.path.append('../')
from data_access.data_access_utility import ABRDataAccess
from classifier.classical_classifiers import SVM, RFClassifier, KNN, LogRegression
from utility.file_utility import FileUtility


feature_lists=[['snps_nonsyn_trimmed'],['gpa_trimmed','gpa_roary'],['genexp'],['snps_nonsyn_trimmed','gpa_trimmed','gpa_roary'],['genexp','gpa_trimmed','gpa_roary'],['genexp','snps_nonsyn_trimmed'],['snps_nonsyn_trimmed','gpa_trimmed','gpa_roary','genexp']]

for cv in ['block','standard']:
    for feature_list in feature_lists:
        ABRAccess=ABRDataAccess('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/intermediate_reps/',feature_list)
        for drug in ABRAccess.BasicDataObj.drugs:
            X_rep, Y, features, final_isolates = ABRAccess.get_xy_prediction_mats(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1})
            FileUtility.save_list('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/classifications_'+cv+'/'+drug+'_'.join(feature_list)+'_isolates',final_isolates)
            print(drug , ' features ',' and '.join(feature_list), ' LR')
            MLRG = LogRegression(X_rep, Y)
            MLRG.tune_and_eval_predefined('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/classifications_'+cv+'/'+drug+'_'.join(feature_list),final_isolates,'../temp_configs/cv3/'+cv+'_cv/'+drug+'_S-vs-R_folds.txt',njobs=40)









