import sys

sys.path.append('../')
from utility.file_utility import FileUtility
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def create_excell_file(input_path, output_path):
    files_cv = FileUtility.recursive_glob(input_path, '*.pickle')

    files_cv.sort()
    table_test = {'classifier': [], 'feature': [], 'CV': [], 'Precision': [], 'Recall': [], 'F1': [],'macroF1': [], 'accuracy': []}
    table_cv = {'classifier': [], 'feature': [], 'CV': [], 'Precision': [], 'Recall': [], 'F1': [], 'macroF1': [],'accuracy': []}

    import warnings
    warnings.filterwarnings('ignore')
    df1=[]
    df2=[]
    for file in files_cv:
        [label_set, conf, label_set, best_score_, best_estimator_,
         cv_results_, best_params_, (cv_predictions_pred, cv_predictions_trues, isolates),
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
        table_test['macroF1'].append(np.round(f1_score(Y_test, Y_test_pred,average='macro'), 2))
        table_test['accuracy'].append(np.round(accuracy_score(Y_test, Y_test_pred), 2))

        table_cv['feature'].append(rep)
        table_cv['classifier'].append(classifier)
        table_cv['CV'].append(CV_scheme)
        table_cv['Precision'].append(np.round(precision_score(cv_predictions_trues, cv_predictions_pred), 2))
        table_cv['Recall'].append(np.round(recall_score(cv_predictions_trues, cv_predictions_pred), 2))
        table_cv['F1'].append(np.round(f1_score(cv_predictions_trues, cv_predictions_pred), 2))
        table_cv['macroF1'].append(np.round(f1_score(cv_predictions_trues, cv_predictions_pred,average='macro'), 2))
        table_cv['accuracy'].append(np.round(accuracy_score(cv_predictions_trues, cv_predictions_pred), 2))
        df1 = pd.DataFrame(data=table_test,
                           columns=['feature', 'CV', 'classifier', 'accuracy', 'Precision', 'Recall', 'F1','macroF1'])
        df2 = pd.DataFrame(data=table_cv,
                           columns=['feature', 'CV', 'classifier', 'accuracy', 'Precision', 'Recall', 'F1','macroF1'])
    writer = pd.ExcelWriter(output_path)
    df1.sort_values(['macroF1','feature','classifier'], ascending=[False, True, True], inplace=True)
    df1.to_excel(writer, 'Test', index=False)

    df2.sort_values(['macroF1','feature','classifier'], ascending=[False, True, True], inplace=True)
    df2.to_excel(writer, 'Cross-validation', index=False)
    writer.save()


def create_excell_project(path, output_path):
    files = FileUtility.recursive_glob(path, '*.xlsx')
    writer = pd.ExcelWriter(output_path+'/classifications.xls', engine='xlsxwriter')


    sheets={'CV std Test':[],'CV std Cross-val':[],'CV tree Test':[],'CV tree Cross-val':[]}
    for file in files:
        phenotype=file.split('/')[-3]
        cv=file.split('/')[-4].split('_')[-2]
        xls=pd.ExcelFile(file)
        df_test=pd.read_excel(xls, 'Test')
        df_cross_val=pd.read_excel(xls,'Cross-validation')

        df_test['phenotype']=phenotype
        if cv=='std':
            sheets['CV std Test'].append(df_test.copy())
        else:
            sheets['CV tree Test'].append(df_test.copy())

        df_cross_val['phenotype']=phenotype
        if cv=='std':
            sheets['CV std Cross-val'].append(df_cross_val.copy())
        else:
            sheets['CV tree Cross-val'].append(df_cross_val.copy())

    for x,frames in sheets.items():
        result = pd.concat(frames).copy()
        result.sort_values(['phenotype', 'macroF1','classifier','feature'], ascending=[True, False, True, True], inplace=True)
        result.to_excel(writer, sheet_name=x, index=False)
    writer.close()
