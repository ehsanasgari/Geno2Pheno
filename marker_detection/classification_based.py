import sys
sys.path.append('../')
from utility.file_utility import FileUtility
from utility.visualization_utility import create_mat_plot
from utility.math_utility import generate_binary
from utility.list_set_util import get_intersection_of_list
import itertools
import numpy as np



files=FileUtility.recursive_glob('/net/sgi/metagenomics/projects/pseudo_genomics/results/geno2pheno_package/K_pneumoniae/feature_selection/classifications/infection_vs_carriage/Human_invasive/','*')
files.sort()

file_list=[]
prev=''
new_list=[]
for file in files:
    if not prev =='':
        new_list=dict([(file.split('_')[-1],file])
        prev=file
    elif ''.join(file.split('_')[0:-1])==''.join(prev.split('_')[0:-1]):
        new_list[file.split('_')[-1]]=file
    else:
        file_list.append(new_list)
        new_list=dict([(file.split('_')[-1],file])
        prev=file

print (new_list)

# features_addresses={'Random Forest':'../../amr_results/5_3_2017/classifications_standard/##drug##gpa_trimmed_gpa_roaryRF_features', 'Chi2':'../../amr_results/res_before_5_3_2017/results/feature_selection/chi2/##drug##gpa_trimmed_gpa_roary','PhyloChi':'/mounts/data/proj/asgari/dissertation/git_repos/amr_results/features/phylochi2/gpa_trimmed_gpa_roary##drug##.txt','SVM':'../../amr_results/features/gpa/##drug##_S-vs-R_non-zero+weights.txt','Treewas':'../../amr_results/features/treewas/gpa/##drug##_S.vs.R_terminal_p-vals_name.txt'}
#
#
# def generate_feature_sets(features_addresses):
#     '''
#         Final all features
#     '''
#     methods=list(features_addresses.keys())
#     methods.sort()
#     res=dict()
#     for drug in ['Ciprofloxacin', 'Tobramycin', 'Colistin', 'Ceftazidim', 'Meropenem']:
#         res[drug]=dict()
#         for idx,method in enumerate(methods):
#             if method=='Random Forest' or method=='SVM':
#                 res[drug][method]=dict([(x.split('\t')[0].replace(' ','').replace('gpa_roary##','').replace('gpa_trimmed##',''),float(x.split('\t')[1])) for x in FileUtility.load_list(features_addresses[method].replace('##drug##',drug))[1::]])
#             if method=='Chi2' or method=='PhyloChi':
#                 res[drug][method]=dict([(x.split('\t')[0].replace(' ','').replace('gain_','').replace('loss_','').replace('gpa_roary##','').replace('gpa_trimmed##',''),float(x.split('\t')[1])) for x in FileUtility.load_list(features_addresses[method].replace('##drug##',drug))[1::] if float(x.split('\t')[1]) > 10 ])
#             if method=='Treewas':
#                 res[drug][method]=dict([(x.split('\t')[0].replace(' ',''),float(x.split('\t')[1])) for x in FileUtility.load_list(features_addresses[method].replace('##drug##',drug))[1::] if float(x.split('\t')[1]) < 0.05 ])
#     return res
#
# def generate_intersection_matrix(features_addresses, drug):
#     '''
#         This function generates intersection table for all methods for a given drug
#     '''
#     methods=list(features_addresses.keys())
#     methods.sort()
#     method_pairs=list(itertools.combinations(range(len(methods)),2))+[(x,x) for x in range(len(methods))]
#     mat=np.zeros((len(methods),len(methods)))
#     for idx,idy in method_pairs:
#         if methods[idx]=='Random Forest' or methods[idx]=='SVM':
#             residx=dict([(x.split('\t')[0].replace(' ','').replace('gpa_roary##','').replace('gpa_trimmed##',''),float(x.split('\t')[1])) for x in FileUtility.load_list(features_addresses[methods[idx]].replace('##drug##',drug))[1::]])
#         if methods[idy]=='Random Forest' or methods[idy]=='SVM':
#             residy=dict([(x.split('\t')[0].replace(' ','').replace('gpa_roary##','').replace('gpa_trimmed##',''),float(x.split('\t')[1])) for x in FileUtility.load_list(features_addresses[methods[idy]].replace('##drug##',drug))[1::]])
#         if methods[idx]=='Chi2' or methods[idx]=='PhyloChi':
#             residx=dict([(x.split('\t')[0].replace(' ','').replace('gain_','').replace('loss_','').replace('gpa_roary##','').replace('gpa_trimmed##',''),float(x.split('\t')[1])) for x in FileUtility.load_list(features_addresses[methods[idx]].replace('##drug##',drug))[1::] if float(x.split('\t')[1]) > 4 ])
#         if methods[idy]=='Chi2' or methods[idy]=='PhyloChi':
#             residy=dict([(x.split('\t')[0].replace(' ','').replace('gain_','').replace('loss_','').replace('gpa_roary##','').replace('gpa_trimmed##',''),float(x.split('\t')[1])) for x in FileUtility.load_list(features_addresses[methods[idy]].replace('##drug##',drug))[1::] if float(x.split('\t')[1]) > 4])
#         if methods[idx]=='Treewas':
#             residx=dict([(x.split('\t')[0].replace(' ',''),float(x.split('\t')[1])) for x in FileUtility.load_list(features_addresses[methods[idx]].replace('##drug##',drug))[1::] if float(x.split('\t')[1]) < 0.05 ])
#         if methods[idy]=='Treewas':
#             residy=dict([(x.split('\t')[0].replace(' ',''),float(x.split('\t')[1])) for x in FileUtility.load_list(features_addresses[methods[idy]].replace('##drug##',drug))[1::] if float(x.split('\t')[1]) < 0.05])
#         mat[idx,idy]=len(set(residx.keys()).intersection(residy.keys()))
#         mat[idy,idx]=mat[idx,idy]
#     return mat, methods
