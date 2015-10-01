#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
DOMAIN ADAPTATION LEARNING OF LINEAR CLASSIFIERS (aka DALC)
See: http://arxiv.org/abs/1506.04573

Executable script to use the classifier (to be used after the learning process).

@author: Pascal Germain -- http://graal.ift.ulaval.ca/pgermain
'''
import common
from dalc import *
from dataset import *
from kernel import *

import sys
import pickle
import argparse 

# common.print_header('CLASSIFICATION')

# Arguments parser   
parser = argparse.ArgumentParser(description="", formatter_class=common.custom_formatter, epilog="")

parser.add_argument("--format", "-f", dest="format",  choices=['matrix', 'svmlight'], default='matrix', help='Datasets format. Default: matrix (each line defines an example, the first column defines the label in {-1, 1}, and the next columns represent the real-valued features)')
parser.add_argument("--model",  "-m", dest="model_file", default='model.bin', help="Model file name. Default: model.bin")
parser.add_argument("--pred",   "-p", dest="prediction_file", default='predictions.out', help="Save predictions into files. Default: predictions.out")

parser.add_argument("test_file", help="Defines the file containing the dataset to classify.")
args = parser.parse_args()

# Main program
###############################################################################
print('... Loading model file ...')
###############################################################################
try:
    with open(args.model_file, 'rb') as model:
        classifier = pickle.load(model)
except:
    print('ERROR: Unable to load model file "' + args.model_file + '".')
    sys.exit(-1)

print('File "' + args.model_file + '" loaded.')

###############################################################################
print('\n... Loading dataset file ...')
###############################################################################
try:
    if args.format == 'matrix':
        test_data = dataset_from_matrix_file(args.test_file)
    elif args.format == 'svmlight':   
        test_data = dataset_from_svmlight_file(args.test_file)
except:
    print('ERROR: Unable to load test file "' + args.test_file + '".')
    sys.exit(-1)
 
print(str(test_data.get_nb_examples()) + ' test examples loaded.')

###############################################################################
print('\n... Prediction ...')
###############################################################################
predictions = classifier.predict(test_data.X)

try:
    predictions.tofile(args.prediction_file, '\n')
    print('File "' + args.prediction_file + '" created.')
except:
    print('ERROR: Unable to write prediction file "' + args.prediction_file + '".')

risk = classifier.calc_risk(test_data.Y, predictions=predictions)

print('Test risk = ' + str(risk))

