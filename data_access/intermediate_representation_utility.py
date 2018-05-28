__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__project__ = "GENO2PHENO of SEQ2GENO2PHENO"
__website__ = ""

import sys
sys.path.append('../')
import codecs
from utility.genotype_file_utility import GenotypeReader
from scipy.sparse import csr_matrix
from utility.file_utility import FileUtility
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


class IntermediateRepCreate(object):
    '''
    This class is written to save
    data for AMR prediction of Pseudomonas Aeruginosa
    '''

    def __init__(self, output_path):
        '''
        :param output_path:
        '''
        print ('data creator..')
        self.output_path=output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    @staticmethod
    def create_feature_files(path):
        '''
        :param path: output path
        :return:
        '''
        base_path='/net/sgi/metagenomics/projects/pseudo_genomics/'

        #### gene_exp
        GenotypeReader.create_read_tabular_file(base_path+'data/gene_expression/v2/rpg_414_log.txt', save_pref=path+'genexp', feature_normalization='zu')
        GenotypeReader.create_read_tabular_file(base_path+'data/gene_expression/v2/rpg_414_log.txt', save_pref=path+'genexp_percent', feature_normalization='percent')

        #### snp
        GenotypeReader.create_read_tabular_file(base_path+'results/featuresAnalysis/v2/non-syn_snps/non_syn_snps_aa_uq.uniq.txt', save_pref=path+'snps_nonsyn_trimmed', feature_normalization='binary')

        #### gpa
        GenotypeReader.create_read_tabular_file(base_path+'results/featuresAnalysis/v2/gpa/annot.uniq.txt', save_pref=path+'gpa_trimmed', feature_normalization='binary')

        #### gpa - roary
        GenotypeReader.create_read_tabular_file(base_path+'results/assembly/v2/roary/v5/out_95/indels/indel_annot.txt', save_pref=path+'gpa_roary', feature_normalization='binary')


        '''
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/snps_nonsyn_trimmed  created successfully containing  426  isolates and  73475  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/snps_nonsyn_envclin_trimmed  created successfully containing  442  isolates and  77748  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/snps_all_envclin_trimmed  created successfully containing  442  isolates and  316168  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/snps_all_full_trimmed  created successfully containing  426  isolates and  306527  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/gpa  created successfully containing  508  isolates and  41872  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/genexp_norm01  created successfully containing  426  isolates and  6026  features
        /mounts/data/proj/asgari/github_data/data/pseudomonas/data_v3/genexp_count  created successfully containing  426  isolates and  6026  features
        '''






if __name__ == "__main__":
    IC=IntermediateRepCreate('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/testingpack/intermediate_rep/')


