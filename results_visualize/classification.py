import sys

sys.path.append('../')
from utility.file_utility import FileUtility
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def create_excell_file(input_path, output_path):
    files_cv = FileUtility.recursive_glob(input_path, '*.pickle')

    files_cv.sort()
    table_test = {'classifier': [], 'feature': [], 'CV': [], 'Precision': [], 'Recall': [], 'F1': [], 'accuracy': []}
    table_cv = {'classifier': [], 'feature': [], 'CV': [], 'Precision': [], 'Recall': [], 'F1': [], 'accuracy': []}

    import warnings
    warnings.filterwarnings('ignore')

    for file in files_cv:
        [label_set, conf, label_set, best_score_, best_estimator_,
         cv_results_, best_params_, (cv_predictions_pred, cv_predictions_trues),
         (Y_test_pred, Y_test)] = FileUtility.load_obj(file)
        rep = file.split('/')[-1].split('_CV_')[0]
        CV_scheme = file.split('_CV_')[1].split('_')[0]
        classifier = file.split('_CV_')[1].split('_')[1].split('.')[0]

        table_test['feature'].append(rep)
        table_test['classifier'].append(classifier)
        table_test['CV'].append(CV_scheme)
        table_test['Precision'].append(np.round(precision_score(Y_test, Y_test_pred), 2))
        table_test['Recall'].append(np.round(recall_score(Y_test, Y_test_pred), 2))
        table_test['F1'].append(np.round(f1_score(Y_test, Y_test_pred), 2))
        table_test['accuracy'].append(np.round(accuracy_score(Y_test, Y_test_pred), 2))

        table_cv['feature'].append(rep)
        table_cv['classifier'].append(classifier)
        table_cv['CV'].append(CV_scheme)
        table_cv['Precision'].append(np.round(precision_score(cv_predictions_trues, cv_predictions_pred), 2))
        table_cv['Recall'].append(np.round(recall_score(cv_predictions_trues, cv_predictions_pred), 2))
        table_cv['F1'].append(np.round(f1_score(cv_predictions_trues, cv_predictions_pred), 2))
        table_cv['accuracy'].append(np.round(accuracy_score(cv_predictions_trues, cv_predictions_pred), 2))
        df1 = pd.DataFrame(data=table_test,
                           columns=['feature', 'CV', 'classifier', 'accuracy', 'Precision', 'Recall', 'F1'])
        df2 = pd.DataFrame(data=table_cv,
                           columns=['feature', 'CV', 'classifier', 'accuracy', 'Precision', 'Recall', 'F1'])
    writer = pd.ExcelWriter(output_path)
    df1.to_excel(writer, 'Test')
    df2.to_excel(writer, 'Cross-validation')
    writer.save()