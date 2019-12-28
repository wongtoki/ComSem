#!/usr/bin/env python
# -*- coding: utf8 -*-

''' Computational semantics course @ RUG-2019
    Lecturer: L.Abzianidze@rug.nl
    Assignment 4: Natural language inference with first-order logic theorem proving

    Usage:
    # Run the prover for the train portion of SICK with verbosity 1
    python3 nli2foli.py --pmb pmb_SICK/ --sick SICK/SICK_trial.txt --sick2pd sick2pd.json -v 1

    # Run the prover for the problem with IDs 1608 2405 from the trial portion of SICK with verbosity 2
    python3 nli2foli.py --pmb pmb_SICK/ --sick SICK/SICK_trial.txt --sick2pd sick2pd.json -v 2 --pids 1608 2405

    # Run the prover for the problem with IDs 1608 from the trial portion of SICK with verbosity 0 and drawing pretty DRS boxes
    python3 nli2foli.py --pmb pmb_SICK/ --sick SICK/SICK_trial.txt --sick2pd sick2pd.json -v 0 --pids 1608 --draw-DRS

'''

#from collections import Counter
import argparse, re
import os
import sys
import logging
from collections import Counter
from logging import debug, info, warning
from os import path as op
import json
import utils
from sick_eval import confusion_matrix_scores
from utils import read_clf, read_sick_problems, sick2pd, read_kb
from clf_referee import get_signature, check_clf, file_to_clfs, pr_box, pr_2rel
from fol import boxes2drs
# from nltk.sem import Expression as semexp
from nltk.inference.tableau import TableauProver
from nltk import Prover9
from knowledge import wn_axioms
# from nltk.inference.prover9 import Prover9
from nltk import ConfusionMatrix
from nltk.sem import Expression as semexp

#################################
def parse_arguments():
    '''Read arguments from a command line'''
    parser = argparse.ArgumentParser(description='Read the SICK dataset files')
    parser.add_argument(
    '--pmb', metavar='DIR', required=True,
        help='Directory containing PMB documents')
    parser.add_argument(
    '--sick', metavar='FILE', required=True,
        help='File containing SICK inference problems')
    parser.add_argument(
    '--pids', metavar='Problem IDs', nargs='+',
        help='IDs of problems that will be solved. Useful for analysis')
    parser.add_argument(
    '--sick2pd', metavar='FILE', default='sick2pd.json',
        help='File containing correspondence between SICK sentence IDs and the PMB documents')
    parser.add_argument(
    '--kb', metavar='FILE',
        help='JSON file containing knowledge base as FOL formulas')
    parser.add_argument(
    '--clf-sig', metavar='FILE', default='clf_signature.yaml',
        help='Signature of CLFs')
    parser.add_argument(
    '--raw2pd', metavar='FILE',
        help='File containing correspondence between SICK raw sentences and the PMB documents')
    parser.add_argument(
    '--out', metavar='FILE',
        help='File where predictions are written')
    parser.add_argument(
    '--draw-DRS', action='store_true',
        help='If true, each DRS is printed graphically. Not suitable for batch inferences')
    parser.add_argument(
    '-v', '--verbose', type=int, default=1,
        help='Verbosity of logging: 0 - warning, 1 - info, 2 - debug')
    args = parser.parse_args()
    # Set verbosity
    verbose = {0:logging.WARNING, 1:logging.INFO, 2:logging.DEBUG}
    logging.basicConfig(format='%(message)s', level=verbose[args.verbose], stream=sys.stdout)
    return args

################################
def pmb2fol(pmb_dir, pd, sig=None, drawDRS=False):
    '''Read a CLF file of the PMB document and convert its content into a first-order logic formula
    '''
    debug("PMB document {}".format(pd))
    clf = read_clf(pmb_dir, pd)
    if not clf: return None
    # Parse clausal forms and read them as a set of connected boxes
    try:
        box_dict, sub_rel, dir_subs, disc_rels, op_types, ops_fine = check_clf(clf, sig)
        if logging.DEBUG >= logging.root.level:
            for b in box_dict: pr_box(box_dict[b])
        debug("sub_rel: {}".format(pr_2rel(sub_rel)))
        debug("dir_sub: {}".format(pr_2rel(dir_subs)))
        debug("Disc rel: {}".format(disc_rels))
    except RuntimeError as e:
        warning("{} has error: {}".format(pd, e))
        return None
    # recover DRS from boxes
    drs = boxes2drs(box_dict, sub_rel, disc_rels)
    if not drs: return None
    if drawDRS: drs.draw()
    # Convert DRS into FOL formula
    fol = drs.fol()
    debug("FOL formula for {}:\n\t{}".format(pd, fol))
    if fol.free():
        warning("The FOL formula of {} has occurrences of free variables: {}".format(pd, fol.free()))
        return None
    return fol

################################
def solve_fol_inference(prem_fol, hypo_fol, str_axioms=[], prover=Prover9(timeout=5)):
    '''Given FOL formulas of a premise and a hypothesis,
       extract relevant knowledge axioms from WordNet, and solve the inference
    '''
    # if one of the formulas is not derived, return neutral
    if not prem_fol or not hypo_fol:
        return 'NEUTRAL', 'No closed formulas'
    # retrieve knowledge from WordNet based on the predicates found in the formulas
    axioms = [ semexp.fromstring(a) for a in str_axioms ]
    axioms = axioms + wn_axioms([prem_fol, hypo_fol])
    if axioms:
        info("Extracted axioms: {}".format(axioms))
    info("Premise Formula: {}".format(prem_fol))
    info("Hypothesis Formula: {}".format(hypo_fol))
    # run prover two times, for checking entailment and contradiction relations
    try:
        Entails = prover.prove(hypo_fol, [prem_fol] + axioms)
        Contradicts = prover.prove(hypo_fol.negate(), [prem_fol] + axioms)
    except Exception as e:
        warning(e)
        return 'NEUTRAL', e
    # if only one of Entails and Contradicts is true, then return that label
    # in all other cases return NEUTRAL,
    # but give a warning if both Entails and Contradicts are true
    if Entails:
        if not Contradicts:
            return 'ENTAILMENT', 'Definite answer'
        warning('Both ENTAILMENT and CONTRADICTION')
        return 'NEUTRAL', 'Mixed answers'
    if Contradicts:
        return 'CONTRADICTION', 'Definite answer'
    return 'NEUTRAL', 'Definite answer'

################################
############## MAIN ############
if __name__ == '__main__':
    args = parse_arguments()
    # Read SICK problems as a lsit of tuples (id, label, premise, hypothesis)
    problems = read_sick_problems(args.sick)
    # Get mapping from SICK problem IDs to a pair of PMB documents for the premise and hypothesis
    sick2pd = sick2pd(problems, mapfile=args.sick2pd)
    # Read the signature of clausal forms as a dictionary
    signature = get_signature(args.clf_sig)
    # read axioms from the knowledge base if file is specified
    kb = read_kb(args.kb) if args.kb else {}
    # keep track of statuses of the theorem proving and predicted labels
    count = Counter()
    gold_labs, pred_labs, prob_ids = [], [], []
    # go through each SICK problem and try to solve it with theorem proving
    for p in problems:
        if args.pids and p[0] not in args.pids:
            continue # skip the rest of the for-loop
        debug("\n\n{:=^100}".format("SICK-"+p[0]))
        info("\nSICK-{0} [{1}]\n\tPrem {4}: {2}\n\tHypo {5}: {3}".format(\
                                                        *p, *sick2pd[p[0]]))
        # get FOL formulas for the corresponding PMB documents (done via building a DRS from a clausal form)
        prem_fol, hypo_fol = [ pmb2fol(args.pmb, pd, sig=signature, drawDRS=args.draw_DRS)\
                               for pd in sick2pd[p[0]] ]
        pred, details = solve_fol_inference(prem_fol, hypo_fol, str_axioms=kb.get(p[0],[]))
        # printing and recording answers
        eureka = "Eureka!!!" if pred == p[1] and pred != 'NEUTRAL' else ''
        info("Result for {:>4}: {} vs {} ({}) {}".format(p[0], p[1], pred.lower(), details, eureka))
        count.update([details])
        # Save only the first letter of the inference relations
        gold_labs.append(p[1][0])
        pred_labs.append(pred[0])
        prob_ids.append(p[0])

    # prints counts of the theorem proving details
    print('{:=^100}'.format(' Status counts ({})'.format(sum(count.values()))))
    for op, cnt in count.most_common():
        print("{}\t{}".format(op, cnt))

    # Draw a confusion matrix
    confusion_matrix_scores(gold_labs, pred_labs, scores=not args.pids)

    # write predictions in the file
    if args.out:
        with open(args.out, 'w') as f:
            mapping = {'N':'NEUTRAL', 'C':'CONTRADICTION', 'E':'ENTAILMENT'}
            for pid, pred in zip(prob_ids, pred_labs):
                f.write("{}:{}\n".format(pid,mapping[pred]))
