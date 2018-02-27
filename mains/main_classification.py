__copyright__ = "Copyright 2017, HH-HZI Project"
__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"

import sys

sys.path.append('../')
from data_access.data_access_utility import ABRDataAccess
from classifier.classical_classifiers import SVM, RFClassifier, KNN


feature_list=['genexp_norm01']
ABRAccess=ABRDataAccess('/mounts/data/proj/asgari/dissertation/datasets/deepbio/pseudomonas/data_v3/',feature_list)
for drug in ABRAccess.BasicDataObj.drugs:
    print(drug)
    X_rep, Y, features, final_isolates = ABRAccess.get_xy_prediction_mats(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1})
    print (drug,' Random Forest ..')
    MRF = RFClassifier(X_rep, Y)
    MRF.tune_and_eval_predefined('results/classification/phylogenetic_snp/'+drug+'_phylogenetic_full_10xfold_RvIS',final_isolates,'/mounts/data/proj/asgari/dissertation/git_repos/amrprediction/data_config/cv/block_cv/'+drug+'_S-vs-R_folds.txt')
    #print (drug,' SVM ..')
    #MSVM = SVM(X_rep, Y)
    #MSVM.tune_and_eval('results/classification/phylogenetic_snp/'+drug+'_phylogenetic_full_10xfold_RvIS')
    #print (drug,' KNN ..')
    #MKNN = KNN(X_rep, Y)
    #MKNN.tune_and_eval('results/classification/phylogenetic_snp/'+drug+'_phylogenetic_full_10xfold_RvIS')
