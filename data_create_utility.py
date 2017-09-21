__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

import codecs
from genotype_file_utility import GenotypeReader

class ABRDataCreate(object):
    '''
    This class is written to save
    data for AMR prediction of Pseudomonas Aeruginosa
    '''

    def __init__(self):
        '''
        To creat the ABD DATA
         self.isolate2label_vec_mapping  ZG02420619 ['1.0', '0.0', '0.0', '', '1.0']
         self.drugs
         self.labeled_isolates: list of sorted isolates
         self.drug2labeledisolates_mapping : {'Ceftazidim': {'BK1496/1': 0, 'BK2061/1': 0, 'CF561_Iso4': 1, 'CF577_Iso7': 1, 'CF590_Iso5': 0.5 ..
        '''

        # init
        self.isolate2label_vec_mapping = dict()
        self.labeled_isolates = list()
        self.drugs = list()
        self.drug2labeled_isolates_mapping = dict()

        # basic loading
        self.make_labels()

    def make_labels(self):
        '''
            This function load labels ZG02420619 ['1.0', '0.0', '0.0', '', '1.0']
        '''
        label_file_address = '/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/mic_bin_without_intermediate.txt'
        rows = [l.replace('\n', '') for l in codecs.open(label_file_address, 'r', 'utf-8').readlines()]
        self.isolate2label_vec_mapping = {
            str(entry.split('\t')[0]): [str(x) for idx, x in enumerate(entry.split('\t')[1::])] for entry in rows[1::]}
        self.labeled_isolates = list(self.isolate2label_vec_mapping)
        self.labeled_isolates.sort()
        self.drugs = rows[0].rstrip().split('\t')[1::]
        # init
        for drug in self.drugs:
            self.drug2labeled_isolates_mapping[drug] = []
        mapping = {'0': 0, '0.0': 0, '1': 1, '1.0': 1, '': 0.5}
        # only consider non-empty values
        for isolate, resist_vec in self.isolate2label_vec_mapping.items():
            for idx, val in enumerate(resist_vec):
                if val in mapping:
                    self.drug2labeled_isolates_mapping[self.drugs[idx]].append((isolate, mapping[val]))
        # generate dict of labels for each class
        for drug in self.drugs:
            self.drug2labeled_isolates_mapping[drug] = dict(self.drug2labeled_isolates_mapping[drug])

    def get_new_labeling(self, mapping={'0': 0, '0.0': 0, '1': 1, '1.0': 1}):
        '''
        Get new labeling
        Load labels
        :param mapping:
        :return:
        '''
        new_drug2labeled_isolates_mapping = dict()
        for drug in self.drugs:
            new_drug2labeled_isolates_mapping[drug] = []
        # only consider non-empty values
        for isolate, resist_vec in self.isolate2label_vec_mapping.items():
            for idx, val in enumerate(resist_vec):
                if val in mapping:
                    new_drug2labeled_isolates_mapping[self.drugs[idx]].append((isolate, mapping[val]))
        return new_drug2labeled_isolates_mapping

    @staticmethod
    def create_feature_files(path):
        '''
        :param path: output path
        :return:
        '''
        base_path='/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/trimmed_features/'
        #### snp
        # non-syn
        GenotypeReader.create_read_tabular_file(base_path+'clinical_snp/non-syn_SNPs_bin_uq.txt.trimmed', save_pref=path+'snps_nonsyn_trimmed', feature_normalization='binary')
        # all clinic_env 316K all
        GenotypeReader.create_read_tabular_file(base_path+'EnviCln_snp.fulltable/all_snps_environ_clinical_bin.trimmed', save_pref=path+'snps_all_envclin_trimmed', feature_normalization='binary', transpose=True)
        # all clinic 306K final
        GenotypeReader.create_read_tabular_file(base_path+'snp.fulltable/all_SNPs_final_bin_uq.txt.trimmed', save_pref=path+'snps_all_full_trimmed', feature_normalization='binary', transpose=True)
        # idk
        GenotypeReader.create_read_tabular_file(base_path+'EnviCln_snp/snp_environ_clinical_bin.txt.trimmed', save_pref=path+'snps_envclin_trimmed', feature_normalization='binary')

        #### gene_presence/absence
        GenotypeReader.create_read_tabular_file('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/annot.txt', save_pref=path+'gpa', feature_normalization='binary')
        #### gene exp.
        GenotypeReader.create_read_tabular_file('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/rpg_log_transformed_426.txt', save_pref=path+'genexp_norm01', feature_normalization='01')
        GenotypeReader.create_read_tabular_file('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/rpg_log_transformed_426.txt', save_pref=path+'genexp_count', feature_normalization='percent')
        '''
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/snps_nonsyn_trimmed  created successfully containing  426  isolates and  73475  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/snps_all_envclin_trimmed  created successfully containing  442  isolates and  316168  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/snps_all_full_trimmed  created successfully containing  426  isolates and  306527  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/gpa  created successfully containing  508  isolates and  41872  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/genexp_norm01  created successfully containing  426  isolates and  6026  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/genexp_count  created successfully containing  426  isolates and  6026  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/snps_envclin_trimmed  created successfully containing  442  isolates and  77748  features
        '''


    @staticmethod
    def get_multilabel_label_dic(isolate2label_vec_mapping):
        '''
            drug resistance profile label
        '''
        mapping = {'': 'I', '0': 'S', '0.0': 'S', '1': 'R', '1.0': 'R'}
        return {k: ''.join([mapping[x] for x in list(v)]) for k, v in isolate2label_vec_mapping.items()}

    @staticmethod
    def get_common_isolates(list_of_list_of_isolates):
        '''
        :param list_of_list_of_isolates:
        :return:
        '''
        common_isolate = set(list_of_list_of_isolates[0])
        for next_list in list_of_list_of_isolates[1::]:
            common_isolate = common_isolate.intersection(next_list)
        common_isolate = list(common_isolate)
        common_isolate.sort()
        return common_isolate



if __name__ == "__main__":
    ABRDataCreate.create_feature_files('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/')
