__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"

from data_access_utility import ABRDataAccess
import numpy as np
from scipy import sparse
from chi2analysis import Chi2Anlaysis
import operator
import codecs
from file_utility import FileUtility

def single_drug_patterns(file_name, list_of_feature=['snps_nonsyn_trimmed' 'snps_all_envclin_trimmed', 'snps_all_full_trimmed' 'snps_nonsyn_envclin_trimmed' 'gpa','genexp_count'], mapping={'0':0,'0.0':0,'1':1,'1.0':1}):
    '''
    :param file_name:
    :param list_of_feature:
    :param mapping:
    :return:
    '''
    ABRAccess=ABRDataAccess('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/',list_of_feature)
    drug2labeledisolates_mapping=ABRAccess.BasicDataObj.get_new_labeling(mapping=mapping)
    global_selected=set()
    for drug in ABRAccess.BasicDataObj.drugs:
        iso2label=drug2labeledisolates_mapping[drug]
        scores=dict()
        for feature_type in list_of_feature:
            print (drug,' ',feature_type)
            iso2label=dict(iso2label)
            idx_removed=[idx for idx, iso in enumerate(ABRAccess.isolates[feature_type]) if iso not in iso2label]
            refined_isos=[iso for iso in ABRAccess.isolates[feature_type] if iso in iso2label]
            Y=[iso2label[iso] for iso in refined_isos]
            X=sparse.csr_matrix(np.delete(ABRAccess.X[feature_type].toarray(),idx_removed, 0))
            CH2=Chi2Anlaysis(X,Y,ABRAccess.feature_names[feature_type])
            sc=CH2.extract_features_fdr('results/features/chi2/separate_drugs/details/'+file_name+'_'+drug+'_'+feature_type+'.txt',500)
            scores[feature_type]=sc
        score_aggregated=[]
        for feature, score_list in scores.items():
            [score_aggregated.append(('##'.join([feature,feat_detail]),value)) for feat_detail,value in score_list]
        score_aggregated=dict(score_aggregated)
        score_aggregated = sorted(score_aggregated.items(), key=operator.itemgetter([1][0]),reverse=True)
        f = codecs.open('results/features/chi2/separate_drugs/aggregation/'+file_name+'_all_'+drug+'.txt','w')
        f.write('\t'.join(['feature', 'score', 'p-value'])+'\n')
        selected=set()
        for w, score in score_aggregated:
            f.write('\t'.join([str(w), str(score[0]), str(score[1])])+'\n')
            if score[0]>10:
                selected.add(str(w))
        f.close()
        if len(selected)>0 and len(global_selected)==0:
            global_selected=selected
        elif len(global_selected)>0:
            global_selected.intersection(selected)
    FileUtility.save_list('esults/features/chi2/separate_drugs/'+file_name+'_alldrugs.txt',list(global_selected))

if __name__ == "__main__":
    single_drug_patterns('R_S')
