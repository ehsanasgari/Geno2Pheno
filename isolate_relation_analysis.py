import numpy as np
from data_create_utility import ABRDataCreate
from math_utility import get_kl_rows
from visualization_utility import create_mat_plot
from drug_relation_analysis import DrugRelation

class IsolatessRelations(object):
    def __init__(self):
        # load ABRDataCreat for basic access
        self.BasicDataObj = ABRDataCreate()
        # init to be filled by make_drug_vector
        self.drug_vectors=[]
        self.drugs=[]
        # fill drugs and drug vectors
        self.make_drug_vector()

    def make_drug_vector(self, mapping={'0': 0, '0.0': 0, '1': 1, '1.0': 1, '': 0.5}):
        '''
        :param mapping: resistance value mapping
        :return: drug vectors
        '''
        self.drug_vectors = np.zeros((len(self.BasicDataObj.drugs), len(self.BasicDataObj.labeled_isolates)))
        for col, isolate in enumerate(self.BasicDataObj.labeled_isolates):
            self.drug_vectors[:, col] = [mapping[res_val] for res_val in
                                         self.BasicDataObj.isolate2label_vec_mapping[isolate]]
        self.drugs = self.BasicDataObj.drugs

    def get_correlation_coefficient(self):
        '''
        :return: Return Pearson product-moment correlation coefficients
        '''
        return np.corrcoef(self.drug_vectors)

    def get_kl_divergence(self):
        '''
        :return: kl-div between drugs
        '''
        return get_kl_rows(self.drug_vectors)

    def get_isolate_profile_kldiv(self):
        '''
        :return: kl_div matrix, list of isolates on col,row
        '''
        return get_kl_rows(self.drug_vectors.T), self.BasicDataObj.labeled_isolates

    def get_isolate_profile_correlation_coefficient(self):
        '''
        :return: corr matrix, list of isolates on col,row
        '''
        return np.corrcoef(self.drug_vectors.T), self.BasicDataObj.labeled_isolates

    def create_kl_divergence(self, filename):
        '''
        :param filename
        to play with colormaps https://matplotlib.org/users/colormaps.html
        '''
        create_mat_plot(self.get_kl_divergence(), self.drugs, 'Drug performance Kullback–Leibler divergence',
                        'results/drug_analysis/' + filename, cmap='Purples')

    def create_correlation_coefficient(self, filename):
        '''
        :param filename
        to play with colormaps https://matplotlib.org/users/colormaps.html
        '''
        create_mat_plot(self.get_correlation_coefficient(), self.drugs,
                        'Drug performance Pearson correlation coefficients', 'results/drug_analysis/' + filename,
                         cmap='Purples')

if __name__ == "__main__":
    DR = DrugRelation()
    DR.create_correlation_coefficient('drugs_corr_SRI')
    DR.create_kl_divergence('drugs_kl_SRI')
