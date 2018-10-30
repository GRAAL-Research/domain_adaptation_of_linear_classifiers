#-*- coding:utf-8 -*-
'''
DOMAIN ADAPTATION OF LINEAR CLASSIFIERS (aka DALC)
See: http://arxiv.org/abs/1506.04573

Dataset class

@author: Pascal Germain -- http://researchers.lille.inria.fr/pgermain/
'''
import numpy as np


class Dataset:
    """ 
    Binary classification dataset
    
    X -- Examples matrix (each row is a vector of features)
    Y -- Label vector
    """
    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y
        
    def load_matrix_file(self, filename, separator=None, first_column_contains_labels=True, last_column_contains_labels=False):       
        """ Load a matrix file, where each line defines an example.
        first_column_contains_labels: specifies that first matrix column defines the labels (default)
        last_column_contains_labels:  specifies that last matrix column defines the labels 
        """
        data_matrix = np.loadtxt(filename, delimiter=separator)
        
        if first_column_contains_labels and last_column_contains_labels:
            raise Exception('first_column_contains_labels and last_column_contains_labels')
        elif last_column_contains_labels:
            self.X = data_matrix[ :, :-1 ]
            self.Y = data_matrix[ :, -1 ]
        elif first_column_contains_labels:
            self.X = data_matrix[ :, 1: ]
            self.Y = data_matrix[ :, 0] 
        else:
            self.X = data_matrix
            self.Y = np.zeros( np.size(data_matrix, 0) )       

    def load_svmlight_file(self, filename, min_features=0):
        """ Load a svmlight file (see http://svmlight.joachims.org/).
        min_features: specifies the minimum number of features for an example
        """
        with open(filename) as f:
            lines_list = f.readlines()
                    
        examples_list = []                  
        labels_list   = []
        nb_features   = min_features
        
        for line in lines_list:
            elems_list = [ e.split(':') for e in line.split() ] 
            
            if len(elems_list[0]) == 1:
                labels_list.append( float(elems_list[0][0]) )
                first_feature_index = 1
            else:
                labels_list.append(0.0)
                first_feature_index = 0
            
            features_list = [ tuple([ int(e[0]), float(e[1]) ])  for e in elems_list[first_feature_index:] ]
            examples_list.append( features_list )
            nb_features = max(nb_features, max( features_list )[0] )
            
        self.X = np.zeros( (len(labels_list), nb_features) )
        for i in range( len(labels_list) ):
            for (j, val) in examples_list[i]:
                self.X[i,j-1] = val
                
        self.Y = np.array(labels_list)
                 
    def get_nb_examples(self):
        """ Return the number of examples of the dataset. """
        if self.X is None: return 0
        return np.size(self.X, 0)       
    
    def get_nb_features(self):
        """ Return the number of features of each example. """
        if self.X is None: return 0
        return np.size(self.X, 1)

    def select_examples(self, indices):
        """ Select the examples of specified indices (and discard others).  """
        self.X = self.X[indices]
        self.Y = self.Y[indices]
        
    def reshape_features(self, new_nb_features):
        """ Add or remove elements to feature vectors. """
        diff_features = new_nb_features - self.get_nb_features()
        if diff_features < 0:
            self.X = self.X[:, :diff_features]        
        elif diff_features > 0:
            nb_examples = self.get_nb_examples()
            self.X = np.hstack(( self.X, np.zeros([nb_examples, diff_features]) ))

        
def dataset_from_matrix_file(filename, separator=None, first_column_contains_labels=True, last_column_contains_labels=False):
    """Utility function. Initialize a dataset and call Dataset.load_matrix_file(...)."""
    data = Dataset()
    data.load_matrix_file(filename, separator, first_column_contains_labels, last_column_contains_labels)
    return data

def dataset_from_svmlight_file(filename, nb_features=None):
    """Utility function. Initialize a dataset and call Dataset.load_svmlight_file(...)."""
    data = Dataset()

    min_features = 0 if nb_features is None else nb_features
    data.load_svmlight_file(filename, min_features)
    
    if nb_features is not None and data.get_nb_features() > nb_features:
        data.reshape_features(nb_features)
    
    return data

