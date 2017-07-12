import codecs
import itertools
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import _pickle as pickle

class ClassifierTuning(object):
    def __init__(self, X, Y, clf_0, parameters):
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
        self.clf = GridSearchCV(estimator=clf_0, param_grid=parameters, cv=cv, n_jobs=40, scoring='f1')
        self.X = X
        self.Y = Y

    def start_validation(self, filename):
        self.clf.fit(self.X, self.Y)
        with open( filename + '.pickle', 'wb') as f:
            pickle.dump(self.clf, f)

def data_preperation():
    data_dir = '/mounts/data/proj/asgari/github_data/data/pseudomonas/data/'
    label_files = 'MIC/v2/mic_bin_with_intermediate.txt'
    gene_expression_file = 'gene_expression/rpg_log_transformed.txt'
    snps_files = 'snp/v2/all_SNPs_final_bin.txt'

    # data reading
    label_mapping = {str(l.strip().split('\t')[0]): [int(float(str(x))) for x in l.strip().split('\t')[1::]] if len(
        l.strip().split('\t')[1::]) == 5 else [0, 0, 0, 0, 0] for l in
                     codecs.open(data_dir + label_files, 'r', 'utf-8').readlines()[1::]}

    # gene expresssion data
    gene_expression = [l.strip() for l in codecs.open(data_dir + gene_expression_file, 'r', 'utf-8').readlines()]
    gene_expression_mapping = {str(entry.split('\t')[0]): [float(str(x)) for x in entry.split('\t')[1::]] for entry in
                               gene_expression[1::]}

    # SNPs  data
    snps = [l.strip().replace(' ', '') for l in codecs.open(data_dir + snps_files, 'r', 'utf-8').readlines()]
    mapping = {'0.0': -1, '0': -1, '1.0': 1, '1': 1, '': 0}
    snps_mapping = {str(entry.split('\t')[0]): [mapping[str(x)] for x in entry.split('\t')[1::]] for entry in snps[1::]}

    # ids
    training_instances = list(
        set(label_mapping.keys()).intersection(gene_expression_mapping.keys()).intersection(snps_mapping.keys()))
    print('Total of training examples ', str(len(training_instances)))

    # concat both data
    X1 = np.array([gene_expression_mapping[x] for x in training_instances])
    X2 = np.array([snps_mapping[x] for x in training_instances])
    X = np.concatenate((X1, X2), axis=1)

    return X, label_mapping, training_instances


if __name__ == '__main__':
    X, Y_map, X_ids = data_preperation()

    param_grid = {"n_estimators": [100, 200, 1000],
                  "criterion": ["gini", "entropy"],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  "max_depth": [10, 20, 100, 500],
                  "min_samples_split": [2, 4, 10, 100],
                  "bootstrap": [True, False]}

    clf_random_forest=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=30,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


    drugs= ['Ciprofloxacin','Tobramycin','Colistin','Ceftazidim','Meropenem']

    for idx, drug in enumerate(drugs):
        print ('start working on '+drug)
        f=codecs.open('RF_'+drug+'.txt','w','utf-8')
        Y=[Y_map[x][idx] for x in X_ids]
        CT=ClassifierTuning(X, Y, clf_random_forest, param_grid)
        CT.start_validation('RF_'+drug)
        f.write(CT.clf.best_score_+'\n')
        f.write(CT.clf.best_params_+'\n')
        f.close()

