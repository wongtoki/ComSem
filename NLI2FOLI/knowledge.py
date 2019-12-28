#!/usr/bin/env python
# -*- coding: utf8 -*-

''' Computational semantics course @ RUG-2019
    Lecturer: L.Abzianidze@rug.nl
    Assignment 4: Natural language inference with first-order logic theorem proving
'''

from __future__ import unicode_literals
import re
from logging import debug, info, warning
from nltk.corpus import wordnet as wn
from nltk.sem import Expression as semexp

#################################
def wn_axioms(formulas):
    '''Get axioms from WordNet for the predicates found in formulas'''
    axioms = set()
    dict_ss_pred = dict_synset_pred(formulas)
    synsets = set(dict_ss_pred.keys())
    hypernym_axioms = get_hypernym_axioms(synsets, dict_ss_pred)
    axioms.update(hypernym_axioms)
    return list(axioms)

#################################
def get_hypernym_axioms(synsets, ss_pred):
    '''Detect hypernym relations among synsets'''
    axioms = set()
    ss_hypers = [ (ss, all_hypernyms(ss)) for ss in synsets ]
    for ss1, hypers1 in ss_hypers:
        for ss2, _ in ss_hypers:
            if ss1 != ss2 and ss2 in hypers1:
                # ss2 is a hypernym of ss1
                axiom = "all x.({}(x) -> {}(x))".format(ss_pred[ss1], ss_pred[ss2])
                axiom = semexp.fromstring(axiom)
                axioms.add(axiom)
    return axioms

#################################
def dict_synset_pred(formulas):
    '''Get synsets of lexical predicates found in the formulas'''
    ss_pred = {}
    for f in formulas:
        for p in f.predicates():
            m = re.match('([a-z]+)_([vnar])(\d\d)$', p.name)
            if m:
                ss = "{}.{}.{}".format(*m.groups())
                try:
                    ss_pred[wn.synset(ss)] = p.name
                except:
                    pass
    return ss_pred

#################################
def all_hypernyms(ss):
    '''Get all hypernyms of a synset'''
    hypernyms = set()
    ancestors = ss.hypernyms()
    while ancestors:
        hypernyms.update(ancestors)
        ancestors = set( g for a in ancestors for g in a.hypernyms() )
    return hypernyms
