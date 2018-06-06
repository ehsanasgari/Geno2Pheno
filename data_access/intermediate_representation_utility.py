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
import tqdm
from scipy import sparse


class IntermediateRepCreate(object):
    '''
    This class is written to save
    the representations
    '''

    def __init__(self, output_path):
        '''
        :param output_path:
        '''
        print ('data creator..')
        self.output_path=output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)


    def create_table(self, path, name, feature_normalization,override=False):

        return GenotypeReader.create_read_tabular_file(path, save_pref=self.output_path+name, feature_normalization=feature_normalization,override=override)

    def create_kmer_table(self, path, k):
        files=FileUtility.recursive_glob(path, '*')
        files.sort()
        strains=[]
        mat=[]
        for file in tqdm.tqdm(files):
            strains.append(file.split('/')[-1].split('.')[0])
            sequences=FileUtility.read_fasta_sequences(file)
            vec,vocab=GenotypeReader.get_nuc_kmer_distribution(sequences,k)
            mat.append(vec)
        mat=sparse.csc_matrix(mat)
        save_path=self.output_path+'sequence_'+str(k)+'mer'
        FileUtility.save_sparse_csr(save_path,mat)
        FileUtility.save_list('_'.join([save_path, 'strains', 'list.txt']), strains)
        FileUtility.save_list('_'.join([save_path, 'features', 'list.txt']), vocab)
        return ('_'.join([save_path])+' created')

if __name__ == "__main__":
    IC=IntermediateRepCreate('/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/testingpack/intermediate_rep/')
    IC.create_table('/net/sgi/metagenomics/projects/pseudo_genomics/results/PackageTesting/K_pneumoniae/genotables/gpa.uniq.mat','uniqGPA','binary')
    IC.create_table('/net/sgi/metagenomics/projects/pseudo_genomics/results/PackageTesting/K_pneumoniae/genotables/non-syn_SNPs.uniq.mat','uniqNonsynSNP','binary')
    IC.create_table('/net/sgi/metagenomics/projects/pseudo_genomics/results/PackageTesting/K_pneumoniae/genotables/syn_SNPs.uniq.mat','uniqNonsynSNP','binary')