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



class GenotypePhenotypeAccess(object):
    '''
        This class is written to handle genotype phenotype access
    '''

    def __init__(self, project_path):
        '''
        To creat the ABD DATA
         self.isolate2label_vec_mapping  ZG02420619 ['1.0', '0.0', '0.0', '', '1.0']
         self.drugs
         self.labeled_isolates: list of sorted isolates
         self.drug2labeledisolates_mapping : {'Ceftazidim': {'BK1496/1': 0, 'BK2061/1': 0, 'CF561_Iso4': 1, 'CF577_Iso7': 1, 'CF590_Iso5': 0.5 ..
        '''

        self.project_path=project_path
        self.metadata_path = self.project_path + '/metadata/'
        self.representation_path = self.project_path + '/intermediate_rep/'
        # init
        self.strain2labelvector = dict()
        self.labeled_strains = list()
        self.phenotypes = list()
        self.phenotype2labeled_strains_mapping = dict()

        # basic loading
        self.make_labels()

    def make_labels(self, pos=['1'], neg=['0']):
        '''
            This function load labels mapping from strain to phenotypes
        '''
        label_file_address =self.metadata_path + 'phenotypes.txt'
        rows = FileUtility.load_list(label_file_address)
        self.strain2labelvector = {
            str(entry.split('\t')[0]): [str(x) for idx, x in enumerate(entry.split('\t')[1::])] for entry in rows[1::]}
        self.labeled_strains = list(self.strain2labelvector)
        self.labeled_strains.sort()
        self.phenotypes = [x for x in rows[0].rstrip().split('\t')[1::]]
        # init
        for phenotype in self.phenotypes:
            self.phenotype2labeled_strains_mapping[phenotype] = []
        mapping = dict([(x,1) for x in pos])
        mapping.update([(x,0) for x in neg])
        # only consider non-empty values
        for strain, phenotype_vec in self.strain2labelvector.items():
            for idx, val in enumerate(phenotype_vec):
                if val in mapping:
                    self.phenotype2labeled_strains_mapping[self.phenotypes[idx]].append((strain, mapping[val]))
        # generate dict of labels for each class
        for phenotype in self.phenotypes:
            self.phenotype2labeled_strains_mapping[phenotype] = dict(self.phenotype2labeled_strains_mapping[phenotype])

    def get_new_labeling(self, pos=['1'], neg=['0']):
        '''
        Get new labeling
        Load labels
        :param mapping:
        :return:
        '''
        mapping = dict([(x,1) for x in pos])
        mapping.update([(x,0) for x in neg])
        new_phenotype2labeled_strain_mapping = dict()
        for drug in self.phenotypes:
            new_phenotype2labeled_strain_mapping[drug] = []
        # only consider non-empty values
        for strain, pheno_vec in self.strain2labelvector.items():
            for idx, val in enumerate(pheno_vec):
                if val in mapping:
                    new_phenotype2labeled_strain_mapping[self.phenotypes[idx]].append((strain, mapping[val]))
        return new_phenotype2labeled_strain_mapping


    def get_multilabel_label_dic(self, mapping = {'0':'F','1':'T'}):
        '''
            Phenotype multilabel
        '''
        return {k: ''.join([mapping[x] for x in list(v)]) for k, v in self.strain2labelvector.items()}

    @staticmethod
    def get_common_isolates(list_of_list_of_strains):
        '''
        :param list_of_list_of_strains:
        :return:
        '''
        common_strains = set(list_of_list_of_strains[0])
        for next_list in list_of_list_of_strains[1::]:
            common_strains = common_strains.intersection(next_list)
        common_strains = list(common_strains)
        common_strains.sort()
        return common_strains
