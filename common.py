#-*- coding:utf-8 -*-
'''
DOMAIN ADAPTATION OF LINEAR CLASSIFIERS (aka DALC)
See: http://arxiv.org/abs/1506.04573

common functions.

@author: Pascal Germain -- http://researchers.lille.inria.fr/pgermain/
'''
import argparse


TEXT_WIDTH = 100
STR_LINE = '-'*TEXT_WIDTH + '\n'


def print_header(subtext):
    text = STR_LINE + \
        'DOMAIN ADAPTATION OF LINEAR CLASSIFIERS (aka DALC) : ' + subtext + '\n' + \
        'Released under the BSD-license\n' + \
        STR_LINE + 'Author: \n' + \
        '    Pascal Germain. Groupe de Recherche en Apprentissage Automatique de l\'Universite Laval (GRAAL).\n\n' + \
        'Reference: \n' + \
        '    Pascal Germain, Amaury Habrard, Francois Laviolette, and Emilie Morvant. \n' + \
        '    A New PAC-Bayesian Perspective on Domain Adaptation.\n' + \
        '    International Conference on Machine Learning (ICML) 2016.\n' + \
        '    http://arxiv.org/abs/1506.04573\n' + \
        STR_LINE

    print(text)


def custom_formatter(prog):
    formatter = argparse.RawDescriptionHelpFormatter(prog, indent_increment=2, max_help_position=20, width=TEXT_WIDTH)
    return formatter
