__copyright__ = "Copyright 2017, HH-HZI Project"
__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"

import sys

sys.path.append('../')
from data_access.data_access_utility import ABRDataAccess
from Bio import Phylo
import numpy as np
import itertools
from multiprocessing import Pool
import tqdm
from utility.file_utility import FileUtility
from utility.featurizer import TextFeature
from chi2analysis.chi2analysis import Chi2Analysis


class PhyloChi2(object):

    def __init__(self, nwk_file="../data_config/mitip_422_gt90.fasttree", feature_list=['snps_nonsyn_trimmed'], load=False):
        '''
            PhyloChi2
        '''
        # data reading
        self.saving_path = '/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/feature_selection/phylochi2/'
        self.resulting_path = '/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/results/feature_selection/phylochi2/'

        self.tree = Phylo.read(nwk_file, "newick")
        ABRAccess = ABRDataAccess(
            '/net/sgi/metagenomics/projects/pseudo_genomics/results/amr_toolkit/intermediate_reps/', feature_list)
        self.X, self.Y, self.features, self.isolates = ABRAccess.get_xy_multidrug_prediction_mats()
        self.drugs = ABRAccess.BasicDataObj.drugs
        self.feature_list=feature_list
        # extract_edges
        if not load:
            self.extract_all_edges()

    def generate_parallel_gainloss_data_for_drug(self, drug_idx, num_p):
        '''
           for  drug_idx using num_p
           generates the data for chi2 in self.gainloss_corpus and self.gainloss_labels
        '''
        triples = []
        for A, B in self.all_edges:
            triples.append((A, B, drug_idx))

        pool = Pool(processes=num_p)

        # prepare dictionary
        gains_losses_corpus = []
        for g_l_c in tqdm.tqdm(pool.imap_unordered(self.get_corpus_labels, triples, chunksize=1),
                               total=len(triples)):
            gains_losses_corpus += g_l_c

        lines = [' '.join(l[1::]) for l in gains_losses_corpus]
        labels = [l[0] for l in gains_losses_corpus]
        TF = TextFeature(lines)
        FileUtility.save_sparse_csr(self.saving_path + '_'.join([self.drugs[drug_idx], 'gainlosses', 'X']),
                                    TF.tf_vec)
        FileUtility.save_list(self.saving_path + '_'.join([self.drugs[drug_idx], 'gainlosses', 'features']),
                              TF.feature_names)
        FileUtility.save_list(self.saving_path + '_'.join([self.drugs[drug_idx], 'gainlosses', 'lables']), labels)

    def extract_all_edges(self):
        '''
            extract all edges
        '''
        terminals = self.tree.get_terminals()
        all_edges = list()
        for t in terminals:
            all_edges.append(PhyloChi2.get_path_edges(self.tree.get_path(t)))
        # check if the edge is meaningful (not having unknown differences)
        all_edges = [(A, B) for A, B in list(itertools.chain(*all_edges)) if
                          len([x for x in B if x in self.isolates]) > 0 and len(
                              [x for x in A if x in self.isolates]) > 0 and (
                          not [x for x in A if x in self.isolates] == [x for x in B if x in self.isolates])]

        temp=[]
        self.all_edges=[]
        for edge in all_edges:
            strx='==>'.join(['###'.join(edge[0]),'###'.join(edge[1])])
            if strx not in temp:
                temp.append(strx)
                self.all_edges.append(edge)

    @staticmethod
    def get_path_edges(node_seq):
        '''
            from node sequence to edges
        '''
        edges = list()
        for first, second in zip(node_seq, node_seq[1:]):
            edges.append(([x.name for x in first.get_terminals()], [x.name for x in second.get_terminals()]))
        return edges

    def get_rep_set_of_nodes(self, A):
        '''
            get representations of nodes
        '''
        idxs = [self.isolates.index(x) for x in A if x in self.isolates]
        res = []
        for arr in self.X[idxs, :].toarray():
            if len(res) == 0:
                res = arr
            else:
                res = np.multiply(res, arr)
        return res

    def get_labels(self, from_labels, to_labels):
        '''
            Rules for chi2 labels
        '''
        if from_labels == to_labels:
            return [0, 0]
        if to_labels == 'R':
            return [2, 0]
        if from_labels == 'R':
            return [0, 2]
        if to_labels == 'I':
            return [1, 0]
        if from_labels == 'I':
            return [0, 1]
        if from_labels == 'S' and (not 'S' in to_labels):
            if 'R' in to_labels:
                return [2, 0]
            else:
                return [1, 0]
        if to_labels == 'S' and (not 'S' in from_labels):
            if 'R' in to_labels:
                return [0, 2]
            else:
                return [0, 1]
        return [0, 0]

    def get_corpus_labels(self, ABDrug_triple):
        '''
            for a single edge betweeb A and B and for drug drug_idx it produces the gaines and losses and phenotype change
        '''
        gains_losses_corpus = []
        A, B, drug_idx = ABDrug_triple
        A = [iso for iso in set(A) if iso in self.isolates]
        B = [iso for iso in set(B) if iso in self.isolates]
        x_parent = self.get_rep_set_of_nodes(A)
        x_self = self.get_rep_set_of_nodes(B)
        x_siblings = self.get_rep_set_of_nodes(list(set(A) - set(B)))
        gain = ['gain_' + self.features[idx] for idx in list(np.where((x_self - x_parent) > 0)[0])]
        loss = ['loss_' + self.features[idx] for idx in list(np.where((x_self - x_siblings) < 0)[0])]
        # extract labels
        sibling_labels = list(set([self.Y[self.isolates.index(iso)][drug_idx] for iso in set(A) - set(B)]))
        self_labels = list(set([self.Y[self.isolates.index(iso)][drug_idx] for iso in B]))
        sibling_labels.sort()
        self_labels.sort()
        label = '=>'.join([''.join(sibling_labels), ''.join(self_labels)])
        temp = []
        if len(gain) > 0:
            temp += gain
        if len(loss) > 0:
            temp += loss
        if len(temp) > 0:
            gains_losses_corpus.append([label] + temp)
        return gains_losses_corpus

    def generate_features_chi2(self):
        '''
        Generate chi2 selected features over the edges and store them for each separate drug
        :return:
        '''
        for drug_idx in range(0, 5):
            print(self.drugs[drug_idx])
            X = FileUtility.load_sparse_csr(self.saving_path + '_'.join([self.drugs[drug_idx], 'gainlosses', 'X.npz']))
            features = FileUtility.load_list(
                self.saving_path + '_'.join([self.drugs[drug_idx], 'gainlosses', 'features']))
            labels = FileUtility.load_list(self.saving_path + '_'.join([self.drugs[drug_idx], 'gainlosses', 'lables']))
            label_map = {'S=>R': 1, 'S=>I': 0, 'S=>IR': 0, 'I=>R': 0, 'IS=>R': 0}
            # label_map={'S=>R':2,'S=>I':1,'S=>IR':2,'I=>R':2,'IR=>I':1,'IR=>R':2,'IS=>R':2,'IS=>I':1,'RS=>I':1,'RS=>R':2,'IRS=>I':1,'IRS=>R':2}
            # label_map={'S=>R':2,'S=>I':1,'S=>IR':2,'I=>R':2,'IS=>R':2}
            row_values = [label_map[x] if x in label_map else 0 for x in labels]
            two_idxs = [idx for idx, x in enumerate(row_values) if x == 2]
            chi2_label = [1 if x > 0 else 0 for x in row_values]
            # X=X.toarray()
            # X[two_idxs,:]=X[two_idxs,:]*2
            #X=csr_matrix(X)
            CHI2 = Chi2Analysis(X, chi2_label, feature_names=features)
            CHI2.extract_features_fdr(self.resulting_path + '_'.join(self.feature_list) + self.drugs[drug_idx] + '.txt', N=100)


if __name__ == "__main__":
    PCh2 = PhyloChi2(feature_list=['gpa_trimmed', 'gpa_roary'],load=True)
    for i in range(0, 5):
        PCh2.generate_parallel_gainloss_data_for_drug(i, 20)
    PCh2.generate_features_chi2()
