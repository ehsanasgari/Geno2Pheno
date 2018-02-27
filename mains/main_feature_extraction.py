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
from chi2analysis.chi2analysis import Chi2Analysis

feature_lists=[['snps_nonsyn_trimmed'],['gpa_trimmed','gpa_roary'],['genexp_percent']]

for feature_list in feature_lists:
    ABRAccess=ABRDataAccess('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/intermediate_reps/',feature_list)
    for drug in ABRAccess.BasicDataObj.drugs:
        print(drug , ' features ',' and '.join(feature_list), ' Random Forest ')
        X_rep, Y, features, final_isolates = ABRAccess.get_xy_prediction_mats(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1})
        CHI2=Chi2Analysis(X_rep, Y, features)
        CHI2.extract_features_fdr('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/feature_selection/chi2/'+drug++'_'.join(feature_list),-1)

