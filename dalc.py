#-*- coding:utf-8 -*-
'''
DOMAIN ADAPTATION OF LINEAR CLASSIFIERS (aka DALC)
See: http://arxiv.org/abs/1506.04573

Learning algorithm implementation

@author: Pascal Germain -- http://researchers.lille.inria.fr/pgermain/
'''
from kernel import Kernel, KernelClassifier
import numpy as np
from math import sqrt, pi
from numpy import exp, maximum
from scipy.special import erf
from scipy import optimize
from collections import OrderedDict

# Some useful constants
CTE_1_SQRT_2    = 1.0 / sqrt(2.0)
CTE_1_SQRT_2PI  = 1.0 / sqrt(2 * pi)
CTE_SQRT_2_PI   = sqrt(2.0 / pi)

# Some useful functions, and their derivatives
def gaussian_loss(x):
    return 0.5 * ( 1.0 - erf(x * CTE_1_SQRT_2) )

def gaussian_loss_derivative(x):
    return -CTE_1_SQRT_2PI * exp(-0.5 * x**2)

def gaussian_convex_loss(x):
    return maximum( 0.5*(1.0-erf(x*CTE_1_SQRT_2)) , -x*CTE_1_SQRT_2PI+0.5 )

def gaussian_convex_loss_derivative(x):
    x = maximum(x, 0.0)
    return -CTE_1_SQRT_2PI * exp(-0.5 * x**2)

def gaussian_disagreement(x):
    return 0.5 * ( 1.0 - (erf(x * CTE_1_SQRT_2))**2 )

def gaussian_disagreement_derivative(x):
    return -CTE_SQRT_2_PI * erf(x * CTE_1_SQRT_2) * exp(-0.5 * x**2)

def gaussian_joint_error(x):
    return 0.25 * ( 1.0 - erf(x * CTE_1_SQRT_2) )**2

def gaussian_joint_error_derivative(x):
    return -CTE_1_SQRT_2PI * exp(-0.5 * x**2)  * ( 1.0 - erf(x * CTE_1_SQRT_2) )

JE_SADDLE_POINT_X = -0.5060544689891808
JE_SADDLE_POINT_Y = gaussian_joint_error(JE_SADDLE_POINT_X)
JE_SADDLE_POINT_DX = gaussian_joint_error_derivative(JE_SADDLE_POINT_X)

def gaussian_joint_error_convex(x):
    return maximum( gaussian_joint_error(x), JE_SADDLE_POINT_DX * (x - JE_SADDLE_POINT_X) + JE_SADDLE_POINT_Y)

def gaussian_joint_error_convex_derivative(x):
    return gaussian_joint_error_derivative( maximum(x, JE_SADDLE_POINT_X) )


# Main learning algorithm
class Dalc:
    def __init__(self, B=1.0, C=1.0, convexify=False, nb_restarts=1, verbose=False):
        """Pbda learning algorithm.
        B: Trade-off parameter 'B' (source joint error modifier)
        C: Trade-off parameter 'C' (target disagreement modifier)
        convexify: If True, the source loss function is convexified (False by default)
        nb_restarts: Number of random restarts of the optimization process.
        verbose: If True, output informations. Otherwise, stay quiet.
        """       
        self.B = float(B)
        self.C = float(C)
        
        self.nb_restarts = int(nb_restarts)
        self.verbose = bool(verbose)

        if convexify:
            self.source_loss_fct = gaussian_joint_error_convex
            self.source_loss_derivative_fct = gaussian_joint_error_convex_derivative
        else:
            self.source_loss_fct = gaussian_joint_error
            self.source_loss_derivative_fct = gaussian_joint_error_derivative

    def learn(self, source_data, target_data, kernel=None, return_kernel_matrix=False):
        """Launch learning process."""
        if kernel is None: kernel = 'linear'

        if type(kernel) is str:
            kernel = Kernel(kernel_str=kernel)

        if self.verbose: print('Building kernel matrix.')
        data_matrix = np.vstack((source_data.X, target_data.X))
        label_vector = np.hstack((source_data.Y, np.zeros(target_data.get_nb_examples())))
        
        kernel_matrix = kernel.create_matrix(data_matrix)
        
        alpha_vector = self.learn_on_kernel_matrix(kernel_matrix, label_vector)

        classifier = KernelClassifier(kernel, data_matrix, alpha_vector)
        if return_kernel_matrix:
            return classifier, kernel_matrix
        else:
            return classifier

    def learn_on_kernel_matrix(self, kernel_matrix, label_vector):  
        """Launch learning process, from a kernel matrix. In label_vector, 0 indicates target examples."""                     
        self.kernel_matrix = kernel_matrix
        self.label_vector = np.array(label_vector, dtype=int)
        self.target_mask = np.array(self.label_vector == 0, dtype=int)
        self.source_mask = np.array(self.label_vector != 0, dtype=int)

        self.nb_examples = len(self.label_vector)
                
        if np.shape(kernel_matrix) != (self.nb_examples, self.nb_examples):
            raise Exception("kernel_matrix and label_vector size differ.")
        
        self.margin_factor = (self.label_vector + self.target_mask) / np.sqrt( np.diag(self.kernel_matrix) )
        
        initial_vector = self.label_vector / float(self.nb_examples)        
        best_cost, best_output = self.perform_one_optimization(initial_vector, 0)
        
        for i in range(1, self.nb_restarts):
            initial_vector = (np.random.rand(self.nb_examples) - 0.5) / self.nb_examples
            cost, optimizer_output = self.perform_one_optimization(initial_vector, i)
            
            if cost < best_cost:
                best_cost = cost
                best_output = optimizer_output        
        
        self.optimizer_output = best_output
        self.alpha_vector = best_output[0]       
        return self.alpha_vector
    
    def perform_one_optimization(self, initial_vector, i):
        """Perform a optimization round."""  
        if self.verbose: print('Performing optimization #' + str(i+1) + '.')
        
        optimizer_output = optimize.fmin_l_bfgs_b(self.calc_cost, initial_vector, self.calc_gradient) 
        cost = optimizer_output[1] 
        
        if self.verbose:
            print('cost value: ' + str(cost))
            for (key, val) in optimizer_output[2].items():
                if key is not 'grad': print(str(key) + ': ' + str(val))                    
    
        return cost, optimizer_output
                           
    def calc_cost(self, alpha_vector, full_output=False):
        """Compute the cost function value at alpha_vector."""
        kernel_matrix_dot_alpha_vector = np.dot(self.kernel_matrix, alpha_vector)
        margin_vector = kernel_matrix_dot_alpha_vector * self.margin_factor
        
        joint_err_vector = self.source_loss_fct(margin_vector) * self.source_mask
        loss_source = joint_err_vector.sum()

        disagreement_vector = gaussian_disagreement(margin_vector) * self.target_mask
        loss_target = disagreement_vector.sum()

        KL = np.dot(kernel_matrix_dot_alpha_vector, alpha_vector) / 2
               
        cost = loss_source / self.C + loss_target / self.B + KL / (self.B * self.C)

        if full_output:
            return cost, loss_source, loss_target, KL
        else:
            return cost
        
    def calc_gradient(self, alpha_vector):
        """Compute the cost function gradient at alpha_vector."""
        kernel_matrix_dot_alpha_vector = np.dot( self.kernel_matrix, alpha_vector )
        margin_vector = kernel_matrix_dot_alpha_vector * self.margin_factor

        d_joint_err_vector = self.source_loss_derivative_fct(margin_vector) * self.margin_factor * self.source_mask
        d_loss_source_vector = np.dot(d_joint_err_vector, self.kernel_matrix)
                        
        d_dis_vector = gaussian_disagreement_derivative(margin_vector) * self.margin_factor * self.target_mask
        d_loss_target_vector = np.dot(d_dis_vector, self.kernel_matrix)
                          
        d_KL_vector = kernel_matrix_dot_alpha_vector

        return d_loss_source_vector / self.C + d_loss_target_vector / self.B + d_KL_vector / (self.B * self.C)
        
    def get_stats(self, alpha_vector=None):
        """Compute some statistics."""
        if alpha_vector is None: alpha_vector = self.alpha_vector
        cost, loss_source, loss_target, KL = self.calc_cost(alpha_vector, full_output=True)
        nb_examples_source = int(np.sum(self.source_mask))

        stats = OrderedDict()

        stats['B'] = self.B
        stats['C'] = self.C
        stats['cost value'] = cost
        stats['loss source'] = loss_source / nb_examples_source
        stats['loss target'] = loss_target / self.nb_examples
        stats['source loss fct'] = self.source_loss_fct.__name__
        stats['KL'] = KL
        stats['optimizer warnflag'] = self.optimizer_output[2]['warnflag']

        return stats

