#-*- coding:utf-8 -*-
'''
DOMAIN ADAPTATION OF LINEAR CLASSIFIERS (aka DALC)
See: http://arxiv.org/abs/1506.04573

common functions.

@author: Pascal Germain -- http://graal.ift.ulaval.ca/pgermain
'''
import argparse

STR_VERSION_NUMBER = '0.90'
STR_VERSION_DATE   = 'November 2, 2015'

TEXT_WIDTH = 100
STR_LINE = '-'*TEXT_WIDTH + '\n'

def print_header(subtext):
    text = STR_LINE + \
        'DOMAIN ADAPTATION OF LINEAR CLASSIFIERS (aka DALC) : ' + subtext + '\n' + \
        'Version ' + STR_VERSION_NUMBER + ' (' + STR_VERSION_DATE + '), Released under the BSD-license\n' + \
        STR_LINE + 'Author: \n' + \
        '    Pascal Germain. Groupe de Recherche en Apprentissage Automatique de l\'Universite Laval (GRAAL).\n\n' + \
        'Reference: \n' + \
        '    Pascal Germain, Amaury Habrard, Francois Laviolette, and Emilie Morvant. \n' + \
        '    A New PAC-Bayesian Perspective on Domain Adaptation.\n' + \
        '    http://arxiv.org/abs/1506.04573\n' + \
        STR_LINE

    print(text)

def custom_formatter(prog):
    formatter = argparse.RawDescriptionHelpFormatter(prog, indent_increment=2, max_help_position=20, width=TEXT_WIDTH)
    return formatter