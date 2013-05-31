#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
PAC-BAYESIAN DOMAIN ADAPTATION (aka PBDA)
Executable script to launch the learning algorithm

@author: Pascal Germain -- http://graal.ift.ulaval.ca/pgermain
'''
import common
from pbda import *
from dataset import *
from kernel import *

import sys
import pickle
import argparse 

common.print_header('LEARNING ALGORITHM')

# Arguments parser   
parser = argparse.ArgumentParser(description="", formatter_class=common.custom_formatter, epilog="")

parser.add_argument("-c", dest="C_value", type=float, default=1.0, help="Trade-off parameter \"C\" (source risk modifier). Default: 1.0")
parser.add_argument("-a", dest="A_value", type=float, default=1.0, help="Trade-off parameter \"A\" (domain disagreement modifier). Default: 1.0")

parser.add_argument("--kernel", "-k", dest="kernel", default="linear", choices=['rbf', 'linear'], help="Kernel function. Default: linear.")
parser.add_argument("--gamma",  "-g", dest="gamma",  type=float, default=1.0, help="Gamma parameter of the RBF kernel. Only used if --kernel is set to rbf. Default: 1.0")
parser.add_argument("--nb_restarts", "-n", dest="nb_restarts",  type=int, default=1, help='Number of random restarts of the optimization process. Default: 1')
parser.add_argument("--format", "-f", dest="format",  choices=['matrix', 'svmlight'], default='matrix', help='Datasets format. Default: matrix (each line defines an example, the first column defines the label in {-1, 1}, and the next columns represent the real-valued features)')
parser.add_argument("--model",  "-m", dest="model_file", default='model.bin', help="Model file name. Default: model.bin")
parser.add_argument("--weight", "-w", dest="weight_file", default='', help="Weight vector file name. Default: (none)")

parser.add_argument("source_file", help="Defines the file containing the source dataset.")
parser.add_argument("target_file", help="Defines the file containing the target dataset.")
args = parser.parse_args()

# Main program
print('... Loading dataset files ...')
if args.format == 'matrix':
    source_data = dataset_from_matrix_file(args.source_file)
    print(str(source_data.get_nb_examples()) + ' source examples loaded.')
    target_data = dataset_from_matrix_file(args.target_file)
elif args.format == 'svmlight':    
    source_data = dataset_from_svmlight_file(args.source_file)
    print(str(source_data.get_nb_examples()) + ' source examples loaded.')
    target_data = dataset_from_svmlight_file(args.target_file, source_data.get_nb_features())
    source_data.reshape_features(target_data.get_nb_features())
print(str(target_data.get_nb_examples()) + ' target examples loaded.')
    
print('\n... Learning ...')
if args.kernel == 'rbf':
    kernel = Kernel('rbf', gamma=args.gamma)
elif args.kernel == 'linear':
    kernel = Kernel('linear')
    
algo = Pbda(A=args.A_value, C=args.C_value, verbose=True, nb_restarts=args.nb_restarts )
classifier = algo.learn(source_data, target_data, kernel) 

print('\n... Saving model: "' + args.model_file + '" ...')
with open(args.model_file, 'wb') as model:
    pickle.dump(classifier, model, pickle.HIGHEST_PROTOCOL)

print('File "' + args.model_file + '" created.')

if len(args.weight_file) > 0:
    classifier.write_to_file(args.weight_file)
    print('File "' + args.weight_file + '" created.')

print('\n... Computing statistics ...')
stats_dict = algo.get_stats()

for key,val in stats_dict.iteritems():
    print( str(key) + ' = ' + str(val) )


