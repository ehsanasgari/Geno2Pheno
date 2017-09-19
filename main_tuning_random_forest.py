from cross_validation import KFoldCrossVal
from data_access_utility import ABRAccessUtility
from sklearn.svm import LinearSVC

clf_svm = LinearSVC(C=1, tol=1e-06, fit_intercept=True, dual=False, penalty='l2')
svm_parameters = [
    {'C': [0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 0.7, 1],
     "class_weight": [None]}]


def tune_eval_all_features(classifier, parameters, features, classifier_name, mapping_name,
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
    ABRAccess = ABRAccessUtility('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/', features)
    for drug in ABRAccess.drugs:
        print(drug)
        X, Y, features = ABRAccess.getXYPredictionMat(drug, mapping=mapping)
        KC = KFoldCrossVal(X, Y, folds=folds, random_state=1, opt_f1_class=opt_f1_class)
        KC.tune_and_evaluate(classifier, parameters, score='opt-f1', n_jobs=n_jobs,
                             file_name='results/tuning/' + classifier_name + '/features/' + mapping_name + '_'.join(features) +'_' + str(folds) + 'fold_' + drug)


def tune_eval_kmers(classifier, parameters, features, classifier_name, k, mapping_name,
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
    for feat in features:
        ABRAccess = ABRAccessUtility(
            '/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/', [feat])
        for drug in ABRAccess.drugs:
            print(feat, drug)
            X, Y, features = ABRAccess.getXYPredictionMat_scaffolds(drug, mapping=mapping)
            KC = KFoldCrossVal(X, Y, folds=folds, random_state=1, opt_f1_class=opt_f1_class)
            KC.tune_and_evaluate(classifier, parameters, score='opt-f1', n_jobs=n_jobs,
                                 file_name='results/tuning/' + classifier_name + '/kmer/' + str(
                                     k) + 'mer/' + mapping_name + '_' + str(k) + 'mer_' + str(folds) + 'fold_' + drug)


tune_eval_all_features(clf_svm, svm_parameters, ['genePA','geneexp','SNPs'], 'svm', 'R_V', folds=3)
#tune_eval_kmers(clf_svm, svm_parameters, ['seq_9_gram_w_idf'], 'svm', 9, 'R_V', folds=3)
#tune_eval_kmers(clf_svm, svm_parameters, ['seq_6_gram'], 'svm', 6, 'R_V', folds=3)
