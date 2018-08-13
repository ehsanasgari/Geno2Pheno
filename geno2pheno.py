#! /usr/bin/python

# -*- coding: utf-8 -*-
__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__project__ = "GENO2PHENO of SEQ2GENO2PHENO"
__website__ = ""

import argparse
import os
import os.path
import shutil
import sys
from xml.dom import minidom
from data_access.intermediate_representation_utility import IntermediateRepCreate
from data_access.data_access import GenotypePhenotypeAccess
from utility.file_utility import FileUtility
from classifier.classical_classifiers import SVM, RFClassifier, KNN, LogRegression
import tqdm

class Geno2Pheno:
    def __init__(self, genml_path, override=1, cores=4):
        '''
            Geno2Pheno commandline use
        '''
        print('Geno2Pheno of Seq2Geno2Pheno 1.0.0')
        self.genml_path=genml_path
        self.override=override
        self.cores=cores
        self.read_data()
        self.predict_block()

    def read_data(self):

        self.xmldoc = minidom.parse(self.genml_path)

        # parse project part
        self.project = self.xmldoc.getElementsByTagName('project')
        self.output = self.project[0].attributes['output'].value
        self.project_name = self.project[0].attributes['name'].value

        if self.override and os.path.exists(self.output):
            var = input("Delete existing files at the output path? (y/n)")
            if var == 'y':
                shutil.rmtree(self.output)
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        log_file = self.output + '/' + 'logfile'
        log_info = ['Project ' + self.project_name]


        self.representation_path = self.output + '/intermediate_rep/'
        IC = IntermediateRepCreate(self.representation_path)

        # load tables
        tabless = self.xmldoc.getElementsByTagName('tables')
        for tables in tabless:
            path = tables.attributes['path'].value
            normalization = tables.attributes['normalization'].value
            prefix = tables.firstChild.nodeValue.strip() + '_'
            if len(prefix) == 1:
                prefix = ''
            for file in FileUtility.recursive_glob(path, '*'):
                log=IC.create_table(file, prefix + file.split('/')[-1], normalization, self.override)
                log_info.append(log)

        tables = self.xmldoc.getElementsByTagName('table')
        for table in tables:
            path = table.attributes['path'].value
            normalization = table.attributes['normalization'].value
            prefix = tables.firstChild.nodeValue.strip() + '_'
            if len(prefix) == 1:
                prefix = ''
            log=IC.create_table(path, prefix + path.split('/')[-1], normalization, self.override)
            log_info.append(log)

        # load sequences
        sequences = self.xmldoc.getElementsByTagName('sequence')
        for sequence in sequences:
            path = sequence.attributes['path'].value
            kmer = int(sequence.attributes['kmer'].value)
            log=IC.create_kmer_table(path,kmer,cores=min(self.cores,4),override=self.override)
            log_info.append(log)

        ## Adding metadata
        metadata_path = self.output + '/metadata/'
        if not os.path.exists(metadata_path):
            os.makedirs(metadata_path)
        # phenotype
        phenotype = self.xmldoc.getElementsByTagName('phenotype')
        if not os.path.exists(metadata_path + 'phenotypes.txt') or self.override:
            FileUtility.save_list(metadata_path + 'phenotypes.txt',
                                  FileUtility.load_list(phenotype[0].attributes['path'].value))

        # tree
        phylogentictree = self.xmldoc.getElementsByTagName('phylogentictree')
        if not os.path.exists(metadata_path + 'phylogentictree.txt') or self.override:
            FileUtility.save_list(metadata_path + 'phylogentictree.txt',
                                  FileUtility.load_list(phylogentictree[0].attributes['path'].value))
        FileUtility.save_list(log_file, log_info)

    def predict_block(self):
        '''
        :return:
        '''
        predict_blocks = self.xmldoc.getElementsByTagName('predict')

        # iterate over predict block
        for predict in predict_blocks:
            predict_path=self.output+'/classifications/'
            # Sub prediction
            FileUtility.ensure_dir(predict_path)
            setting_name=predict.attributes['name'].value
            subdir=predict_path+setting_name+'/'

            FileUtility.ensure_dir(subdir)

            ## label mapping
            labels=predict.getElementsByTagName('labels')[0].getElementsByTagName('label')
            mapping=dict()
            for label in labels:
                val=label.attributes['value'].value
                phenotype=label.firstChild.nodeValue.strip()
                mapping[phenotype]=int(val)

            ## optimizing for ..
            optimization=predict.getElementsByTagName('optimize')[0].firstChild.nodeValue.strip()
            ## number of folds
            self.cvbasis=predict.getElementsByTagName('eval')[0].firstChild.nodeValue.strip()
            folds=int(predict.getElementsByTagName('eval')[0].attributes['folds'].value)
            test_ratio=float(predict.getElementsByTagName('eval')[0].attributes['test'].value)

            if optimization not in ['accuracy','scores_r_1','scores_f1_1','scores_f1_0','f1_macro','f1_micro']:
                print ('Error in choosing optimization score')

            ## Genotype tables
            GPA=GenotypePhenotypeAccess(self.output)
            ## iterate over phenotypes if there exist more than one
            for phenotype in GPA.phenotypes:
                print ('working on phenotype ',phenotype)
                FileUtility.ensure_dir(subdir+phenotype+'/')

                ## create cross-validation
                FileUtility.ensure_dir(subdir+phenotype+'/cv/')
                if self.cvbasis=='tree':
                    FileUtility.ensure_dir(subdir+phenotype+'/cv/tree/')
                    if self.override or not FileUtility.exists(subdir+phenotype+'/cv/tree/'+''.join([phenotype,'_',setting_name,'_folds.txt'])):
                        GPA.create_treefold(subdir+phenotype+'/cv/tree/'+''.join([phenotype,'_',setting_name,'_folds.txt']), folds, test_ratio, phenotype, mapping)
                else:
                    FileUtility.ensure_dir(subdir+phenotype+'/cv/rand/')
                    if self.override or not FileUtility.exists(subdir+phenotype+'/cv/rand/'+''.join([phenotype,'_',setting_name,'_folds.txt'])):
                        GPA.create_randfold(subdir+phenotype+'/cv/rand/'+''.join([phenotype,'_',setting_name,'_folds.txt']), folds, test_ratio, phenotype, mapping)

                features=[x.split('/')[-1].replace('_feature_vect.npz','') for x in FileUtility.recursive_glob(self.representation_path, '*.npz')]
                ## iterate over feature sets
                for feature in features:
                    classifiers=[]
                    for model in predict.getElementsByTagName('model'):
                        for x in model.childNodes:
                            if not x.nodeName=="#text":
                                classifiers.append(x.nodeName)
                    X, Y, feature_names, final_strains = GPA.get_xy_prediction_mats([feature], phenotype, mapping)

                    ## iterate over classifiers
                    for classifier in tqdm.tqdm(classifiers):
                        if classifier.lower()=='svm':
                            Model = SVM(X, Y)
                            Model.tune_and_eval_predefined(subdir+phenotype+'/'+'_'.join([feature]),njobs=self.cores, kfold=folds, feature_names=feature_names)
                        if classifier.lower()=='rf':
                            Model = RFClassifier(X, Y)
                            Model.tune_and_eval_predefined(subdir+phenotype+'/'+'_'.join([feature]),njobs=self.cores, kfold=folds, feature_names=feature_names)
                        if classifier.lower()=='lr':
                            Model = LogRegression(X, Y)
                            Model.tune_and_eval_predefined(subdir+phenotype+'/'+'_'.join([feature]),njobs=self.cores, kfold=folds, feature_names=feature_names)

                        #if classifier.lower()=='dnn':
                        #    Model = DNN(X, Y)
                        #    Model.tune_and_eval(subdir+phenotype+'/'+'_'.join([feature]),njobs=self.cores, kfold=10)





#elif mode=='pairs':
#
#            prefix_list=[list(x) for x in list(itertools.combinations(prefix_list,2))]
#        #elif mode=='multi':
#        #else:


def checkArgs(args):
    '''
        This function checks the input arguments and returns the errors (if exist) otherwise reads the parameters
    '''
    # keep all errors
    err = "";
    # Using the argument parser in case of -h or wrong usage the correct argument usage
    # will be prompted
    parser = argparse.ArgumentParser()

    # parse #################################################################################################
    parser.add_argument('--genoparse', action='store', dest='genml_path', default=False, type=str,
                        help='GENML file to be parsed')

    parser.add_argument('--override', action='store', dest='override',default=1, type=int,
                        help='Override the existing files?')
    # required='--genoparse' in sys.argv

    parser.add_argument('--cores', action='store', dest='cores',default=4, type=int,
                        help='Number of cores to be used')


    parsedArgs = parser.parse_args()

    if (not os.access(parsedArgs.genml_path, os.F_OK)):
        err = err + "\nError: Permission denied or could not find the labels!"
        return err
    G2P = Geno2Pheno(parsedArgs.genml_path, parsedArgs.override, parsedArgs.cores)
    return False

if __name__ == '__main__':
    err = checkArgs(sys.argv)
    if err:
        print(err)
        exit()
