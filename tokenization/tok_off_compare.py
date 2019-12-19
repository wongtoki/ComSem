#!/usr/bin/env python
# -*- coding: utf8 -*-

# The evaluation code for the segmentation assignment (week 1)
# of Computational Semantics Course @ University of Groningen
# Disclaimer: the code comes WITHOUT ANY WARRANTY :) email me for any unexpected behaviour
# contact: L.Abzianidze@rug.nl

import argparse
import codecs
import re
import shutil
import os
import os.path as op 
import sys
from collections import Counter
from string import maketrans

#################################
def parse_arguments():
    '''read arguments'''
    parser = argparse.ArgumentParser(description='Take two en.tok.off files or the whole Part directory (with your en.tok.off files there) and compare them based on TOIS annotation')
    parser.add_argument('-d', '--dir', help='Path to the Part directory, e.g., ./p51')
    parser.add_argument('-f', '--files', nargs=2, 
        help='original tok.off file and new tok.off file, e.g, en.tok.off en.tok.off.s1234567')
    parser.add_argument('-r', '--rawfile', default='', 
        help='raw file name, e.g., en.raw (Default "en.raw" in the same directory as the first file)')
    parser.add_argument('-m', '--myfile', default='en.tok.off.my', 
        help='name of your produced en.tok.off file, e.g, en.tok.off.s1234567 (Default "en.tok.off.my")')
    parser.add_argument('-v', '--verbose', default=0, type=int, 
        help='Use verbosity with levels: 1, 2, 3 (Default 0). If no verbosity, the Progress Counter is shown')
    parser.add_argument('-b', '--beam', default=120, type=int, 
        help='Beam size for printing TOIS annotation in the terminal (Default 120)')
    args = parser.parse_args()
    return args

############## counter as a progress bar #################
def counter_prog(i):
    '''Displays a counter to show the progress'''
    sys.stdout.write('\r{}:'.format(i))
    sys.stdout.flush()

############## tok.off & raw into TOIS #############
def off2TOIS(raw_path, off_path, verbose=0):
    '''From en.tok.off file and en.raw obtain the TOIS annotation'''
    (T, O, I, S) = 'TOIS'
    raw_file = codecs.open(raw_path, 'r', "utf-8")
    off_file = codecs.open(off_path, 'r', "utf-8") 
    raw = raw_file.read()
    raw_file.close()
    raw = raw.rstrip()
    # read tok.off into list of tuples
    off_toks = [ re.match('(\d+) (\d+) (\d+)(\d{3}) (.+)\n$', line).groups() for line in off_file ]
    off_file.close()
    off_toks = [ (int(start), int(end), int(tok_id), tok) for (start, end, sen_id, tok_id, tok) in off_toks ]
    tok_i = 0
    tok_char_i = 0
    tois = ''
    # scans the characters of 'raw' and checks if those appear in the tokens of off_toks
    for (char_i, char) in enumerate(raw):
        if verbose >= 4: print 'off2TOIS: {} at {}; {}th token list element with details {} and inner char position {};'.format(
                                    char, char_i, tok_i, off_toks[tok_i], tok_char_i),   
        try:
            # throws IndexError when char follows the last token and is outside the token, 
            # e.g., when raw is '"it is over."' and " has O in '."'
            (start, end, tok_id, tok) = off_toks[tok_i]
        except IndexError:
            tois += O
            if verbose >= 4: print 'gets {} label'.format(tois[-1]) 
            continue
        try:    
            # throws IndexError when char follows a token and is outside this token, 
            # e.g., when raw is 'the- dog' and '-' has O in 'the' 
            is_part_of_token =  (char_i in range(start, end)) and (tok[tok_char_i] == char)
        except IndexError:
            is_part_of_token = False
        if is_part_of_token: # char is in the offset range and occurs in the token
            if tok_char_i == 0: #start of token
                if tok_id == 1: # start of sentence
                    tois += S
                else:
                    tois += T
            else:
                tois += I
            tok_char_i += 1
        else:
            tois += O 
        if char_i == end - 1: #char's position is the last in the token's offset
            tok_i += 1
            tok_char_i = 0
        if verbose >= 4: print 'gets {} label'.format(tois[-1])  
    #make one line 'raw' for aligning annotations and raw
    one_line_raw = re.sub('[\r\n]', ' ', raw) 
    return (tois, one_line_raw)   

############## Draw confusion matrix #############
def draw_matrix(matrix):
    '''Draw a confusion matrix'''
    TOIS = 'TOIS'
    title = 'ORIG\\MINE' 
    w = len(str(sum(matrix.values()))) #max width for numbers
    rcell = '{:>' + str(w+2) + '}' #right aligned cell
    left_cell = '{:^' + str(len(title)) + '}' #centered cell
    header = rcell * len(TOIS) 
    #print header
    print '{pad} Confusion Matrix {pad}'.format(**{'pad':'='*12})
    print title + header.format(*TOIS) #draw ----- line
    print ('-' * len(title)) * (len(TOIS) + 1)
    for i in TOIS:
        sys.stdout.write( left_cell.format(i) )
        for j in TOIS:
            sys.stdout.write( rcell.format(matrix[(i,j)]) )
        print ''

############## compare two TOIS annotations #############
def TOIS_matrix(tois1, tois2):
    '''Build a confusion matrix of two TOIS annotations.
       Return: the matrix, weather the annotations are different, diff of the annotations
    '''
    matrix = Counter()
    compare = '' # recods the positions of differences
    for (i, a) in enumerate(tois1):
        matrix.update([(a, tois2[i])])
        compare += '$' if a != tois2[i] else ' '
    different = '$' in compare # dollar $ indicates difference
    return (matrix, different, compare) 

############## compare two en.tok.off files #############
def compare_tok_off(raw, file1, file2, n=100, verbose=0):
    '''compares two tok.off files using raw file. 
       Prints differences and returns teh confusion matrix 
    '''
    (tois1, line_raw) = off2TOIS(raw, file1, verbose=verbose)
    (tois2, line_raw) = off2TOIS(raw, file2, verbose=verbose) 
    # throw error is length of annotations or raw file is different
    if len(tois1) != len(tois2) or len(tois1) != len(line_raw):
        raise RuntimeError('Two TOIS annotations or RAW have different lengths')
    (matrix, different, compare) = TOIS_matrix(tois1, tois2) 
    # can report different annotations 
    if ((verbose >= 1) and different) or (verbose >= 2):
        # instead of TOIS labels adopts more readbale labels
        mapping = maketrans("TOIS", "^ -#")
        tois1 = tois1.translate(mapping)
        tois2 = tois2.translate(mapping)
        for i in range(0, len(tois1), n): 
            print u'1LINE_RAW:{}\nTOIS_ORIG:{}\nTOIS_MINE:{}\nSHOW_DIFF:{}'.format(
                line_raw[i:i+n], tois1[i:i+n], tois2[i:i+n], compare[i:i+n] ) 
    return matrix

############## MAIN #############
if __name__ == '__main__':
    args = parse_arguments()
    count = 0 # progress counter
    matrix = Counter() # confusion matrix
    if args.dir is not None:
    # scans Part directory and documents inside 
        for (root, dirs, files) in os.walk(args.dir):
            count += 1
            if not args.verbose: counter_prog(count)
            f_num = len([i for i in files if i in ['en.tok.off', args.myfile]])
            if f_num < 2: # there is only one en.tok.off file that matches specified names in teh directory
                continue
            if f_num > 2: # more than 2 en.tok.off files in the directory
                if args.verbose: print "Don't know which files to compare in", root
                continue
            else:
            # compares two en.tok,off files, also using raw file 
                raw_path = op.join(root, 'en.raw') 
                off_path = op.join(root, 'en.tok.off')  
                myoff_path = op.join(root, args.myfile)  
                if args.verbose >= 3: print 'Working with', raw_path, off_path, myoff_path  
                matrix += compare_tok_off(raw_path, off_path, myoff_path, n=args.beam, verbose=args.verbose)
        if not args.verbose: print ' Done!'
        draw_matrix(matrix) 
    elif args.files is not None:
    # when two specific en.tok.off files are given for comparison 
        (off_path, myoff_path) = args.files
        raw_path = args.rawfile if args.rawfile else op.join(op.dirname(off_path), 'en.raw')
        matrix = compare_tok_off(raw_path, off_path, myoff_path, n=args.beam, verbose=args.verbose)   
        draw_matrix(matrix)    
    else:
        raise argparse.ArgumentError('Specify either files or a directory') 
        




