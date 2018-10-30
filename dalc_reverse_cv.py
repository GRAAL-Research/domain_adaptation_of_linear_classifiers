#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
DOMAIN ADAPTATION OF LINEAR CLASSIFIERS (aka DALC)
See: http://arxiv.org/abs/1506.04573

Executable script to perform reverse cross-validation

@author: Pascal Germain -- http://researchers.lille.inria.fr/pgermain/
'''
import common
from dalc import *
from dataset import *
from kernel import *

import sys
import argparse 

from math import floor
import numpy as np

common.print_header('REVERSE CROSS-VALIDATION HELPER')

# Arguments parser
parser = argparse.ArgumentParser(description="", formatter_class=common.custom_formatter, epilog="")

parser.add_argument("-b", dest="B_value", type=float, default=1.0, help="Trade-off parameter \"B\" (source joint error modifier). Default: 1.0")
parser.add_argument("-c", dest="C_value", type=float, default=1.0, help="Trade-off parameter \"C\" (target disagreement modifier). Default: 1.0")
parser.add_argument("--nb_folds", "-cv", dest="nb_folds",  type=int, default=5, help='Number of cross validation folds. Default: 5')

parser.add_argument("--kernel", "-k", dest="kernel", default="linear", choices=['rbf', 'linear'], help="Kernel function. Default: linear.")
parser.add_argument("--gamma",  "-g", dest="gamma",  type=float, default=1.0, help="Gamma parameter of the RBF kernel. Only used if --kernel is set to rbf. Default: 1.0")
parser.add_argument("--nb_restarts", "-n", dest="nb_restarts",  type=int, default=1, help='Number of random restarts of the optimization process. Default: 1')
parser.add_argument("--format", "-f", dest="format",  choices=['matrix', 'svmlight'], default='matrix', help='Datasets format. Default: matrix (each line defines an example, the first column defines the label in {-1, 1}, and the next columns represent the real-valued features)')

parser.add_argument("source_file", help="Defines the file containing the source dataset.")
parser.add_argument("target_file", help="Defines the file containing the target dataset.")
args = parser.parse_args()

# Main program
###############################################################################
print('... Loading dataset files ...')
###############################################################################
try:
    if args.format == 'matrix':
        source_data = dataset_from_matrix_file(args.source_file)
    elif args.format == 'svmlight':   
        source_data = dataset_from_svmlight_file(args.source_file)
except:
    print('ERROR: Unable to load source file "' + args.source_file + '".')
    sys.exit(-1)
 
print(str(source_data.get_nb_examples()) + ' source examples loaded.')

try:
    if args.format == 'matrix':
        target_data = dataset_from_matrix_file(args.target_file)
    elif args.format == 'svmlight':
        target_data = dataset_from_svmlight_file(args.target_file, source_data.get_nb_features())
        source_data.reshape_features(target_data.get_nb_features())
except:
    print('ERROR: Unable to load target file "' + args.target_file + '".')
    sys.exit(-1)
 
print(str(target_data.get_nb_examples()) + ' target examples loaded.')  

###############################################################################
print('\n... Building kernel matrix ...')
###############################################################################
if args.kernel == 'rbf':
    kernel = Kernel('rbf', gamma=args.gamma)
elif args.kernel == 'linear':
    kernel = Kernel('linear')

data_matrix  = np.vstack(( source_data.X, target_data.X) )
full_label_vector = np.hstack(( source_data.Y, target_data.Y ))
        
full_kernel_matrix = kernel.create_matrix(data_matrix)

###############################################################################
print('\n... Preparing folds  ...')
###############################################################################
nb_source = source_data.get_nb_examples()
fold_size = float(nb_source)/args.nb_folds        
source_indices_groups = [ np.arange( int(floor(i*fold_size)), int(floor((i+1)*fold_size)) ) for i in range(args.nb_folds) ]

nb_target = target_data.get_nb_examples()
fold_size = float(nb_target)/args.nb_folds        
target_indices_groups = [ np.arange( int(floor(i*fold_size)), int(floor((i+1)*fold_size)) )+nb_source for i in range(args.nb_folds) ]

# Display some information to the user   
row_format = "{:<10}" * 4
print(row_format.format('Fold #', 'source', 'target', 'valid'))
print('-'*40)
for i in range(args.nb_folds):
    print(row_format.format(i+1, nb_source-len(source_indices_groups[i]), nb_target-len(target_indices_groups[i]), len(source_indices_groups[i])))

# Utility function to get validation fold indices
def folds_indices(indices_groups, i_fold): 
    indices = indices_groups[:]
    valid_indices = indices.pop(i_fold)    
    train_indices = np.hstack(indices)
    return train_indices, valid_indices

risk_list = []
for i_fold in range(args.nb_folds):   
###############################################################################
    print('\n... Performing validation round # ' + str(i_fold+1) + '...')
###############################################################################  

    # Build kernel matrices for this validation round
    train_source_indices, valid_source_indices = folds_indices(source_indices_groups, i_fold)
    train_target_indices, _ = folds_indices(source_indices_groups, i_fold)
    
    fold_indices = np.hstack( (train_source_indices, train_target_indices) ) 
    train_kernel_matrix = full_kernel_matrix[np.ix_( fold_indices,  fold_indices )]
    train_label_vector = np.hstack(( full_label_vector[ train_source_indices ], np.zeros(len(train_target_indices)) ))

    valid_kernel_matrix = full_kernel_matrix[np.ix_( valid_source_indices, fold_indices )]
    valid_label_vector = full_label_vector[valid_source_indices]

    # Learn a classifier and assign label to target examples
    print('\n*** Source --> Target ***')
    algo = Dalc(C=args.C_value, B=args.B_value, verbose=True, nb_restarts=args.nb_restarts )
    alpha_vector = algo.learn_on_kernel_matrix(train_kernel_matrix, train_label_vector)
    classifier = KernelClassifier(Kernel('precomputed'), None, alpha_vector)
    
    predicted_label_vector = np.sign( classifier.predict(train_kernel_matrix) )[ len(train_source_indices): ] 
    reverse_label_vector = np.hstack(( np.zeros(len(train_source_indices)), predicted_label_vector))   
    
    # Learn a reverse classifier and test it on the validation fold
    print('\n*** Target --> Source ***')
    algo = Dalc(C=args.C_value, B=args.B_value, verbose=True, nb_restarts=args.nb_restarts )
    alpha_vector = algo.learn_on_kernel_matrix(train_kernel_matrix, reverse_label_vector)            
    classifier = KernelClassifier(Kernel('precomputed'), None, alpha_vector)
        
    valid_risk = classifier.calc_risk(valid_label_vector, valid_kernel_matrix)
    risk_list += [ valid_risk ]
    print('\nFold risk = ' + str(valid_risk))

############################################################################### 
print('\n... Summary ...')
###############################################################################
valid_risk = np.mean(risk_list)

row_format = "{:<10}" * 5
print(row_format.format('Fold #', 'source', 'target', 'valid', 'risk'))
print('-'*55)
for i in range(args.nb_folds):
    print(row_format.format(i+1, nb_source-len(source_indices_groups[i]), nb_target-len(target_indices_groups[i]), len(source_indices_groups[i]), risk_list[i]))

print('\nValidation risk = ' + str(valid_risk))




