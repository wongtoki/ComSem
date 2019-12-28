#!/usr/bin/env python
# -*- coding: utf8 -*-

''' Computational semantics course @ RUG-2019
    Lecturer: L.Abzianidze@rug.nl
    Assignment 4: Natural language inference with first-order logic theorem proving

    Usage:
        python3 sick_eval.py --sick SICK/SICK_train.txt --pred train.ans
        python3 sick_eval.py --sick SICK/SICK_trial.txt --pred trial.ans  --filter CE
'''

import argparse, re
from collections import Counter
import utils
from utils import read_sick_problems
from nltk import ConfusionMatrix

#################################
def parse_arguments():
    '''Read arguments from a command line'''
    parser = argparse.ArgumentParser(description='Read the SICK dataset files')
    parser.add_argument(
    '--sick', metavar='FILE', required=True,
        help='File containing SICK problems with gold inference labels')
    parser.add_argument(
    '--pred', metavar='FILE', required=True,
        help='File containing predictions for SICK problems')
    parser.add_argument(
    '--filter',
        help='Combination of gold and system labels, e.g., "EN"')
    args = parser.parse_args()
    return args

#################################
def read_system_predictions(file_path):
    '''Read file each line containing a problem ID and a prediction delimited with a colon'''
    with open(file_path) as f:
        for line in f:
            m = re.match('(\d+)\:([A-Z]+)', line)
            if m:
                yield (m.group(1), m.group(2))

#################################
def confusion_matrix_scores(gold_labs, pred_labs, scores=True):
    '''Draw a confusion matrix and calculate accuracy, precision and recall scores'''
    # Draw a confusion matrix
    if not gold_labs or not pred_labs:
        raise RuntimeError("One of the prediction lists is empty")
    if len(gold_labs) != len(pred_labs):
        raise RuntimeError("The number of predictions != the number of gold labels")
    cm = ConfusionMatrix(gold_labs, pred_labs)
    print(cm.pretty_format(show_percents=False))

    # calculate and print accuracy, precision and recall for a SICK part (not for individual problems)
    if scores:
        pre = (cm[('E','E')] + cm[('C','C')]) / float(sum([ cm[(i, j)] for i in 'NEC' for j in 'EC' ]))
        rec = (cm[('E','E')] + cm[('C','C')]) / float(sum([ cm[(i, j)] for i in 'EC' for j in 'NEC' ]))
        acc = (cm[('E','E')] + cm[('C','C')] + cm[('N','N')]) / float(cm._total)
        print("Accuracy: {:.2f}%\nPrecision: {:.2f}%\nRecall: {:.2f}%".format(acc*100, pre*100, rec*100))

################################
############## MAIN ############
if __name__ == '__main__':
    args = parse_arguments()
    # read system predictions form the file
    pred = { i:p for i, p in read_system_predictions(args.pred) }
    # Read SICK problems as a lsit of tuples (id, label, premise, hypothesis)
    problems = read_sick_problems(args.sick)
    # create lists of gold and predicted labels for the same problems, contract labels to the initial letters
    gold_list, pred_list = [], []
    for p in problems:
        gold_list.append(p[1][0])
        pred_list.append(pred[p[0]][0])
        # print SICk problems with a particular gold and system labels
        if args.filter and gold_list[-1] + pred_list[-1] == args.filter:
            print("SICK-{}   {}\n{}\n{}\n".format(*p))
    confusion_matrix_scores(gold_list, pred_list)
