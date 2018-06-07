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
from utility.file_utility import FileUtility

class Geno2Pheno:
    def __init__(self, genml_path, override=1, cores=4):
        '''
            Geno2Pheno commandline use
        '''
        print('Geno2Pheno of Seq2Geno2Pheno 1.0.0')

        xmldoc = minidom.parse(genml_path)

        # parse project part
        project = xmldoc.getElementsByTagName('project')
        output = project[0].attributes['output'].value
        project_name = project[0].attributes['name'].value

        if override and os.path.exists(output):
            var = input("Delete existing files at the output path? (y/n)")
            if var == 'y':
                shutil.rmtree(output)
        if not os.path.exists(output):
            os.makedirs(output)

        log_file = output + '/' + 'logfile'
        log_info = ['Project ' + project_name]


        representation_path = output + '/intermediate_rep/'
        IC = IntermediateRepCreate(representation_path)

        # load tables
        tabless = xmldoc.getElementsByTagName('tables')
        for tables in tabless:
            path = tables.attributes['path'].value
            normalization = tables.attributes['normalization'].value
            prefix = tables.firstChild.nodeValue.strip() + '_'
            if len(prefix) == 1:
                prefix = ''
            for file in FileUtility.recursive_glob(path, '*'):
                log=IC.create_table(file, prefix + file.split('/')[-1], normalization, override)
                log_info.append(log)

        tables = xmldoc.getElementsByTagName('table')
        for table in tables:
            path = table.attributes['path'].value
            normalization = table.attributes['normalization'].value
            prefix = tables.firstChild.nodeValue.strip() + '_'
            if len(prefix) == 1:
                prefix = ''
            log=IC.create_table(path, prefix + path.split('/')[-1], normalization, override)
            log_info.append(log)

        # load sequences
        sequences = xmldoc.getElementsByTagName('sequence')
        for sequence in sequences:
            path = sequence.attributes['path'].value
            kmer = int(sequence.attributes['kmer'].value)
            log=IC.create_kmer_table(path,kmer,cores=cores)
            log_info.append(log)

        ## Adding metadata
        metadata_path = output + '/metadata/'
        if not os.path.exists(metadata_path):
            os.makedirs(metadata_path)
        # phenotype
        phenotype = xmldoc.getElementsByTagName('phenotype')
        if not os.path.exists(metadata_path + 'phenotypes.txt') or override:
            FileUtility.save_list(metadata_path + 'phenotypes.txt',
                                  FileUtility.load_list(phenotype[0].attributes['path'].value))

        # tree
        phylogentictree = xmldoc.getElementsByTagName('phylogentictree')
        if not os.path.exists(metadata_path + 'phylogentictree.txt') or override:
            FileUtility.save_list(metadata_path + 'phylogentictree.txt',
                                  FileUtility.load_list(phylogentictree[0].attributes['path'].value))
        FileUtility.save_list(log_file, log_info)


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
                        help='Override the existing files?', required='--genoparse' in sys.argv)

    parser.add_argument('--cores', action='store', dest='cores',default=4, type=int,
                        help='Number of cores to be used')

    # svm ######################################################################################################

    parsedArgs = parser.parse_args()

    if (not os.access(parsedArgs.genml_path, os.F_OK)):
        err = err + "\nError: Permission denied or could not find the labels!"
        return err

    if parsedArgs.genoparse:
        G2P = Geno2Pheno(parsedArgs.genml_path, parsedArgs.override, parsedArgs.cores)


    return False


if __name__ == '__main__':
    err = checkArgs(sys.argv)
    if err:
        print(err)
        exit()
