#!/usr/bin/env python
# -*- coding: utf8 -*-

''' Computational semantics course @ RUG-2019
    Lecturer: L.Abzianidze@rug.nl
    Assignment 4: Natural language inference with first-order logic theorem proving
'''

#from collections import Counter
import argparse, re
import os
from collections import defaultdict
import logging
from logging import debug, info, warning
from os import path as op
import json
import sys
from clf_referee import file_to_clfs


#################################
def counter_prog(i, step=100):
    '''Displays a counter to show the progress'''
    if i % 100 == 0:
        sys.stdout.write('\r{}:'.format(i))
        sys.stdout.flush()
    return i + 1

################################
def read_raw(raw_file):
    '''Read SICK problem sentences from a raw file of a PMB document
    '''
    with open(raw_file) as f:
        raw = f.read().strip()
        if not raw:
            warning('The document has an empty raw: ' + raw_file)
    return raw

################################
def read_clf(pmb_dir, pd):
    '''Read CLF file of the part/doc from the PMB directory
    '''
    # Find a CLF file
    for d in ['silver', 'bronze', 'gold']:
        clf_path = op.join(pmb_dir, 'data', d, pd, 'en.drs.clf')
        if op.isfile(clf_path):
            break # stop as the CLF file is found
    try:
        [clf], _ = file_to_clfs(clf_path)
        #debug(clf)
        return clf
    except FileNotFoundError:
        warning("No CLF file for {}".format(pd))
        return None

################################
def sick_raw_pd(pmb_dir, mapfile=None):
    '''Create a dictionary with a SICK raw sentence as a key and a PMB part/document as a value
    '''
    # If the SICK sentence and pmb doc correspondence is provided, read it from the file
    if mapfile and op.isfile(mapfile):
        with open(mapfile) as jsonfile:
            return json.load(jsonfile)
    # Otherwise read the correspondence from the pmb directory
    raw2pd = dict()
    counter = 0
    # scan the pmb doc directory and read the SICK sentence id and pmb doc correspondence
    for root, dirs, files in os.walk(pmb_dir):
        pd = re.search('p\d{2}/d\d{4}', root)
        if not pd:
            # Don't scan the directory if it doesn't correspond to a PMB document
            continue
        if 'en.raw' in files:
            raw_file = op.join(root, 'en.raw')
            raw = read_raw(raw_file)
            if raw in raw2pd:
                warning('Raw /{}/ was already found in a doc {}'.format(raw, raw2pd[raw]))
            raw2pd[raw] = pd.group()
            # show a progress of scanning
            counter += 1
            #if counter % 100 == 0: print('.', end='')
        else:
            warning('No raw file in: ' + root)
    info("{} SICK sentences were read from {} PMB docs".format(len(raw2pd), counter))
    # export the correspondence info as JSON for future re-use
    if mapfile:
        with open(mapfile, 'w') as jsonfile:
            json.dump(raw2pd, jsonfile)
    return raw2pd

################################
def read_kb(kb_file):
    '''Read knowledge base from the file
    '''
    with open(kb_file) as jsonfile:
        kb = json.load(jsonfile)
    return kb

################################
def sick2pd(problems, mapfile=None, raw2pd=None):
    '''Create a dictionary with SICK problem IDs as keys and
       a pair of PMB part/documents (for premise and hypothesis) as values.
    '''
    # If the SICK sentence IDs and pmb doc correspondence is provided, read it from the file
    if mapfile and op.isfile(mapfile):
        with open(mapfile) as jsonfile:
            sick2pd = json.load(jsonfile)
            sick2pd = { p[0] : sick2pd[p[0]] for p in problems }
    else:
        # Otherwise read the correspondence from the pmb directory
        sick2pd = dict()
        counter = 0
        # scan the pmb doc directory and read the SICK sentence id and pmb doc correspondence
        for pid, _, prem, hypo in problems:
            try:
                sick2pd[pid] = [raw2pd[prem], raw2pd[hypo]]
            except KeyError as e:
                warning(e)
        # export the correspondence info as JSON for future re-use
        if mapfile:
            with open(mapfile, 'w') as jsonfile:
                json.dump(sick2pd, jsonfile)
    info("PMB part/documents read for {} problems".format(len(sick2pd)))
    return sick2pd

################################
def read_sick_problems(sick_file):
    '''read SICk problems from the SICK_part file as a list:
       [(id, gold_label, {'p':premise, 'h':hypothesis}), ...]
    '''
    with_answers = True # train and trial parts come with answers, while test doesn't
    with open(sick_file) as f:
        problems = []
        for p in f:
            # ignore the non-problem lines
            if not re.match('\d+\t', p):
                continue
            # parse the problem
            problem_parts = re.split('\s*\t\s*', p.strip())
            if len(problem_parts) == 5:
                (pid, prem, hypo, _, label) = problem_parts
            else:
                (pid, prem, hypo, label) = problem_parts + [None]
            problems.append((pid, label, prem, hypo))
        info("{} SICK problems read from {}".format(len(problems), sick_file))
    return problems
