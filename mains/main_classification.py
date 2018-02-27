__copyright__ = "Copyright 2017-2018, HH-HZI"
__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"


import sys

sys.path.append('../')
from data_access.data_access_utility import ABRDataAccess
from classifier.classical_classifiers import SVM, RFClassifier, KNN


cvs=['block','standard']
cv=cvs[1]
feature_lists=[['snps_nonsyn_trimmed'],['gpa_trimmed','gpa_roary'],['genexp']]

errors=[]
for feature_list in feature_lists:
    ABRAccess=ABRDataAccess('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/intermediate_reps/',feature_list)
    for drug in ABRAccess.BasicDataObj.drugs:
        print(drug , ' features ',' and '.join(feature_list), ' Random Forest ')
        X_rep, Y, features, final_isolates = ABRAccess.get_xy_prediction_mats(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1})
        MRF = RFClassifier(X_rep, Y)
        MRF.tune_and_eval_predefined('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/classifications_'+cv+'/'+drug+'_'.join(feature_list),final_isolates,'../data_config/cv/'+cv+'_cv/'+drug+'_S-vs-R_folds.txt', feature_names=features)
        print(drug , ' features ',' and '.join(feature_list), ' KNN')
        MKNN = KNN(X_rep, Y)
        MKNN.tune_and_eval_predefined('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/classifications_'+cv+'/'+drug+'_'.join(feature_list),final_isolates,'../data_config/cv/'+cv+'_cv/'+drug+'_S-vs-R_folds.txt')
        print(drug , ' features ',' and '.join(feature_list), ' SVM')
        MSVM = SVM(X_rep, Y)
        MSVM.tune_and_eval('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/classifications_'+cv+'/'+drug+'_'.join(feature_list),final_isolates,'../data_config/cv/'+cv+'_cv/'+drug+'_S-vs-R_folds.txt')

feature_list=['snps_nonsyn_trimmed','gpa_trimmed','gpa_roary','genexp']
ABRAccess=ABRDataAccess('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/intermediate_reps/',feature_list)
for drug in ABRAccess.BasicDataObj.drugs:
    print(drug , ' features ',' and '.join(feature_list), ' Random Forest ')
    X_rep, Y, features, final_isolates = ABRAccess.get_xy_prediction_mats(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1})
    MRF = RFClassifier(X_rep, Y)
    MRF.tune_and_eval_predefined('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/classifications_'+cv+'/'+drug+'_'.join(feature_list),final_isolates,'../data_config/cv/'+cv+'_cv/'+drug+'_S-vs-R_folds.txt', feature_names=features)
    print(drug , ' features ',' and '.join(feature_list), ' KNN')
    MKNN = KNN(X_rep, Y)
    MKNN.tune_and_eval_predefined('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/classifications_'+cv+'/'+drug+'_'.join(feature_list),final_isolates,'../data_config/cv/'+cv+'_cv/'+drug+'_S-vs-R_folds.txt')
    print(drug , ' features ',' and '.join(feature_list), ' SVM')
    MSVM = SVM(X_rep, Y)
    MSVM.tune_and_eval('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/classifications_'+cv+'/'+drug+'_'.join(feature_list),final_isolates,'../data_config/cv/'+cv+'_cv/'+drug+'_S-vs-R_folds.txt')

