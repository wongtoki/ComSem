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
    extra_hypernym_axioms = get_more_hypernym_axioms(synsets, dict_ss_pred)
    antonym_axioms = get_antonym_axioms(synsets, dict_ss_pred) # synonyms and antonyms

    # if hypernym_axioms:
    #     print("HYP AXIOMS", hypernym_axioms)
    # if antonym_axioms:
    #     print("ANTONYMS", antonym_axioms)

    # axioms.update(antonym_axioms)
    axioms.update(hypernym_axioms)

    return list(axioms)


def get_more_hypernym_axioms(synsets, ss_pred):
    """Detect more hypernym relations for each synset pair

    :param li synsets: the synsets returned from the premise and hypothesis formula
    :param dict ss_pred: the dictionary that maps synsets back to their formula form,
        since only the antonyms for these synsets are relevant in this formula.

    :rtype: set()
    :return: synonym axioms for each synsets
    """
    axioms = set()
    for ss1 in synsets:

        if ss1.entailments():
            # denotes how verbs are involved
            entailment_li = ss1.entailments()
            for entailment in entailment_li:
                if entailment in synsets:
                    # Got one: Synset('watch.v.01') Synset('look.v.01')
                    # print(ss1, entailment)
                    # TODO: model this relation, because it actually occurs in the FOL formulas
                    continue

        if ss1.part_meronyms():
            for meronym in ss1.part_meronyms():
                # meronym is a part of ss1 as in "stump" is a part of "tree" or "hand" is a part of "body"
                if meronym in synsets:
                    # only if the meronym is another synset that occurs in the FOL formulas
                    ### TODO: model this relation, because it actually occurs in the FOL formulas
                    ### all meronym is a part of ss1
                    ### smth like this
                    continue

        if len(ss1.member_holonyms()) > 1:
            # ss is a member of each ss.member_holonyms():
            # in the way that "fish" is a part of "pesces" and a "school (of fish)"
            # print("S", ss, ss.member_holonyms())
            # for s in ss.member_holonyms():
            #     print(s.definition())
            # TODO: can we model this relation also in FOL? is pretty much the same as the get_more_hypernym_axioms
            continue

        for ss2 in synsets:
            lowest_hypernym = ss1.lowest_common_hypernyms(ss2)
            if ss1 != ss2 and lowest_hypernym:
                # ss1 and ss2 have a common hypernym (so they are at least somewhat related)
                if lowest_hypernym[0].min_depth() > 5:
                    # it is not a top common hypernym (such as "person", "whole" or "artifact")
                    # ss1 and ss2 share a hypernym indirectly!

                    # if ss1.path_similarity(ss2) > 0.5:
                        # checks if they are similar according to similarity measure of WordNet
                    # print("{0} and {1} in common: {2} ({3})".format(ss1, ss2, ss1.lowest_common_hypernyms(ss2), ss1.path_similarity(ss2)))

                    # Synset('road.n.01') and Synset('street.n.02') in common: [Synset('road.n.01')](0.3333333333333333)
                    ### TODO: how to say thet both ss1 == a type of ss1.lowest_common_hypernyms(ss2) and
                    ### TODO: ss2 == a type of ss1.lowest_common_hypernyms(ss2)
                    ### all x.(Attribute(ss1(x),s) & ss2(s)) ???
                    continue


    return axioms

#################################
def get_antonym_axioms(synsets, ss_pred):
    """Detect antonyms for each synset pair

    :param li synsets: the synsets returned from the premise and hypothesis formula
    :param dict ss_pred: the dictionary that maps synsets back to their formula form,
        since only the antonyms for these synsets are relevant in this formula.

    :rtype: set()
    :return: antonym axioms for each synsets
    """
    axioms = set()
    for ss in synsets:
        for lemma in ss.lemmas():

            pertainyms = lemma.pertainyms()
            for p in pertainyms:
                if p.synset() in synsets:
                    # TODO: this one does not occur, so not model it? Maybe look with Blanca at this
                    continue


            # antonym relations are captured in lemmas, not in synsets
            for antonym_lemma in lemma.antonyms():
                antonym = antonym_lemma.synset() # return lemma form to synset form
                if antonym in ss_pred.keys():
                    # only if the antonym occurs in the set of synsets form the formulas do we append it
                    ### TODO: how do I add the axiom for a contradiction?
                    axiom = "-(some x.({}(x) -> {}(x)))".format(ss_pred[ss], ss_pred[antonym])
                    axiom = semexp.fromstring(axiom)
                    axioms.add(axiom)

    return axioms

#################################
def get_hypernym_axioms(synsets, ss_pred):
    '''Detect hypernym relations among synsets'''
    axioms = set()
    ss_hypers = [ (ss, all_hypernyms(ss)) for ss in synsets ]
    for ss1, hypers1 in ss_hypers:
        for ss2, _ in ss_hypers:
            if ss1 != ss2 and ss2 in hypers1:
                # ss2 is a direct hypernym of ss1
                axiom = "all x.({}(x) -> {}(x))".format(ss_pred[ss1], ss_pred[ss2])
                axiom = semexp.fromstring(axiom)
                axioms.add(axiom)
    return axioms

def all_hypernyms(ss):
    '''Get all hypernyms of a synset'''
    hypernyms = set()
    ancestors = ss.hypernyms()
    while ancestors:
        hypernyms.update(ancestors)
        ancestors = set( g for a in ancestors for g in a.hypernyms() )
    return hypernyms

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

