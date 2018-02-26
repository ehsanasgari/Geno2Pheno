import sys
sys.path.append('../')
from data_access.data_access_utility import ABRDataAccess
from classifier.classical_classifiers import SVM, RFClassifier, KNN


feature_list=['snps_all_full_trimmed','genexp_norm01','gpa']
ABRAccess=ABRDataAccess('/mounts/data/proj/asgari/dissertation/datasets/deepbio/pseudomonas/data_v3/',feature_list)
for drug in ABRAccess.BasicDataObj.drugs:
    print(drug)
    X_rep, Y, features, final_isolates = ABRAccess.get_xy_prediction_mats(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1,'':0})
    print (drug,' Random Forest ..')
    MRF = RFClassifier(X_rep, Y)
    MRF.tune_and_eval('../results/classification/snps/RF/'+drug+'_10xfold_RvIS',feature_names=features)
