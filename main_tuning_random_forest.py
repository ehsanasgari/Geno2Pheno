from cross_validation import KFoldCrossVal
from data_access_utility import ABRDataAccess
from sklearn.ensemble import RandomForestClassifier

clf_RF =RandomForestClassifier(bootstrap=True, criterion='gini',
            min_samples_split= 2, max_features='auto', min_samples_leaf=1, n_estimators=1000, n_jobs=50)

param_grid = {"n_estimators": [100, 200, 500, 1000],
              "criterion": ["gini", "entropy"],
              'max_features': ['auto', 'sqrt'],
             'min_samples_split':[2,5,10],
             'min_samples_leaf':[1,2,5]}

def tune_eval_all_features(classifier, parameters, feature_types, classifier_name, mapping_name,
                    mapping={'0': 0, '0.0': 0, '1': 1, '1.0': 1}, folds=5, opt_f1_class=0, n_jobs=50):
    '''
    :param classifier:
    :param parameters:
    :param features:
    :param classifier_name:
    :param k:
    :param mapping_name:
    :param mapping:
    :param folds:
    :param opt_f1_class:
    :param n_jobs:
    :return:
    '''
    ABRAccess=ABRDataAccess('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/',feature_types)
    for drug in ABRAccess.BasicDataObj.drugs:
        print(drug)
        X, Y, features = ABRAccess.get_xy_prediction_mats(drug, mapping=mapping)
        print ('to be at ','results/tuning/' + classifier_name + '/features/' + mapping_name + '_'.join(feature_types) +'_' + str(folds) + 'fold_' + drug)
        KC = KFoldCrossVal(X, Y, folds=folds, random_state=1, opt_f1_class=opt_f1_class)
        KC.tune_and_evaluate(classifier, parameters, score='f1_macro', n_jobs=n_jobs,
                             file_name='results/tuning/' + classifier_name + '/features/' + mapping_name + '_'.join(feature_types) +'_' + str(folds) + 'fold_' + drug)



settings={'R_S':{'0': 0, '0.0': 0, '1': 1, '1.0': 1},'I_S':{'0':0,'0.0':0,'':1},'R_I':{'1':1,'1.0':1,'':0}, 'R_IS':{'0':0,'0.0':0,'1':1,'1.0':1,'':0},'S_IR':{'0':1,'0.0':1,'1':0,'1.0':0,'':0},'I_RS':{'':1,'0':0,'0.0':0,'1':0,'1.0':0}}

feature_list=['snps_nonsyn_trimmed', 'gpa','genexp_norm01']
# ['snps_all_envclin_trimmed', 'snps_all_full_trimmed','snps_nonsyn_trimmed' ,'snps_nonsyn_envclin_trimmed']

#for k,new_mapping in settings.items():
settings={'R_S':{'0': 0, '0.0': 0, '1': 1, '1.0': 1}}
for k,new_mapping in settings.items():
    tune_eval_all_features(clf_RF, param_grid, feature_list, 'random_forest', 'NALL_mf1_'+k, folds=5, mapping=new_mapping)

