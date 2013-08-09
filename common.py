#-*- coding:utf-8 -*-
'''
PAC-BAYESIAN DOMAIN ADAPTATION (aka PBDA)
common functions.

@author: Pascal Germain -- http://graal.ift.ulaval.ca/pgermain
'''
import argparse

STR_VERSION_NUMBER = '0.901'
STR_VERSION_DATE   = 'August 9, 2013'

TEXT_WIDTH = 100
STR_LINE = '-'*TEXT_WIDTH + '\n'

def print_header(subtext):
    text = STR_LINE + \
        'PAC-BAYESIAN DOMAIN ADAPTATION (aka PBDA) : ' + subtext + '\n' + \
        'Version ' + STR_VERSION_NUMBER + ' (' + STR_VERSION_DATE + '), Released under the BSD-license\n' + \
        'http://graal.ift.ulaval.ca/pbda/ \n' + \
        STR_LINE + 'Author: \n' + \
        '    Pascal Germain. Groupe de Recherche en Apprentissage Automatique de l\'Universite Laval (GRAAL).\n\n' + \
        'Reference: \n' + \
        '    Pascal Germain, Amaury Habrard, Francois Laviolette, and Emilie Morvant. \n' + \
        '    A PAC-Bayesian Approach for Domain Adaptation with Specialization to Linear Classifiers. \n' + \
        '    International Conference on Machine Learning (ICML) 2013. \n' + \
        STR_LINE

    print(text)

def custom_formatter(prog):
    formatter = argparse.RawDescriptionHelpFormatter(prog, indent_increment=2, max_help_position=20, width=TEXT_WIDTH)
    return formatter