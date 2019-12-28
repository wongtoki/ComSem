#!/usr/bin/env python
# -*- coding: utf8 -*-

''' Computational semantics course @ RUG-2019
    Lecturer: L.Abzianidze@rug.nl
    Assignment 4: Natural language inference with first-order logic theorem proving
'''

#from collections import Counter
import logging
from logging import debug, info, warning
from clf_referee import pr_box
import re
# from nltk.sem import Expression as semexp
from clf_referee import pr_box, pr_2rel, Box
from nltk.sem import DrtExpression as drtexp


################################
def boxes2drs(box_dict, sub_rel, disc_rels):
    '''Convert a set of boxes into a DRS
    '''
    # ignore discourse relations with uncovered discourse relations
    if has_dis_rel(['ATTRIBUTION', 'NECESSITY', 'POSSIBILITY'], disc_rels):
        return None
    # Remove tense from boxes and get only direct sub-rels
    box_dict, sub_rel = remove_tense(box_dict, sub_rel)
    dir_subs = dir_sub(sub_rel)
    debug("dir_subs: {}".format(pr_2rel(dir_subs)))
    if logging.DEBUG >= logging.root.level:
        for b in box_dict: pr_box(box_dict[b])
    # There are still IMP and DIS conditions in boxes, remove such explicit sub-rels
    nested = consider_nested_boxes(box_dict, dir_subs)
    # visually independent boxes
    vib = set(box_dict.keys()) - nested
    # squeeze all boxes into one
    box = make_one_box(box_dict, vib, dir_subs, disc_rels)
    if logging.DEBUG >= logging.root.level:
        pr_box(box)
    # convert a box into a drs
    drs_str = box2drs(box, box_dict)
    debug("DRS str: {}".format(drs_str))
    drs = drtexp.fromstring(drs_str)
    debug("DRS expression: {}".format(drs))
    return drs

################################
def make_one_box(box_dict, vib, dir_subs, disc_rels):
    '''Merge all boxes into one box, some boxes are embedded
    '''
    eliminate_negation(box_dict, vib, dir_subs, disc_rels)
    eliminate_implication(box_dict, vib, dir_subs, disc_rels)
    eliminate_disjunction(box_dict, vib, dir_subs, disc_rels)
    if next((r for r, b1, b2 in disc_rels if r not in ['ALTERNATION', 'EXPLANATION', 'CONTINUATION']), False):
        raise RuntimeError('Some other discourse relations remain: {}'.format(disc_rels))
    # merge all boxes safely
    box = merge_all_boxes(box_dict, vib, dir_subs)
    return box

################################
def box2drs(box, box_dict):
    '''Convert a big unified box into a DRS
    '''
    refs_row = ", ".join(list(box.refs))
    cond_row = []
    for c in box.conds:
        if c[0] == 'NEGATION':
            cond_row.append("- {}".format(box2drs(box_dict[c[1]], box_dict)))
        elif c[0] in ['IMP']:
            cond_row.append("({} -> {})".format(box2drs(box_dict[c[1]], box_dict), box2drs(box_dict[c[2]], box_dict)))
        elif c[0] in ['DIS']:
            cond_row.append("({} | {})".format(box2drs(box_dict[c[1]], box_dict), box2drs(box_dict[c[2]], box_dict)))
        else:
            cond_row.append(cond2pred(c))
    cond_row = ', '.join(cond_row)
    return "([{}], [{}])".format(refs_row, cond_row)


###############################################################################
#                            translation to FOL
###############################################################################

################################
def cond2pred(c):
    'If there is an intermediate element between x and y'
    if re.match('"[nvar]\.\d+"', c[1]):
        return "{}_{}{}({})".format(cl2pred(c[0]), c[1][1], c[1][3:-1], arg2term(c[2]))
    if c[0].isupper():
        op2str = {'EQU':'=', 'NEQ':'!='}
        if op2str.get(c[0], False):
            return "({} {} {})".format(arg2term(c[1]), op2str[c[0]], arg2term(c[2]))
        return "{}({}, {})".format(c[0], arg2term(c[1]), arg2term(c[2]))
    if c[0][:2].istitle():
        # remove hyphens from roles
        return "{}({},{})".format(cl2pred(c[0]), arg2term(c[1]), arg2term(c[2]))
    raise RuntimeError("Unknown Condition: {}".format(c))

################################
def arg2term(arg):
    'Convert argument of the clause into a FOL term'
    arg = arg.strip('"')
    char_mapping = {'.':'_', '~':'_'}
    chars = [ char_mapping.get(i, i) for i in arg ]
    arg = ''.join(chars)
    mapping = {'-':'minus', '+':'plus'}
    return mapping.get(arg, arg)

################################
def cl2pred(cl):
    'Convert clause operator to a predicate'
    char_mapping = {'-':'_', '~':'_'}
    chars = [ char_mapping.get(i, i) for i in cl ]
    return ''.join(chars)

###############################################################################
#                            Operations of Boxes
###############################################################################

################################
def merge_all_boxes(box_dict, vib, dir_subs):
    'Merge all boxes and modify relations accordingly'
    # If there is only one box remianed then vib has it explicitly
    box = Box('final_box')
    for b in vib:
        box1 = box_dict.pop(b)
        box.refs.update(box1.refs)
        box.conds.update(box1.conds)
        box.cond_refs.update(box1.cond_refs)
    return box

def consider_nested_boxes(box_dict, dir_subs):
    '''Sometimes IMP and DIS are left in conditions due to incomplete translation.
       Make such relations implicit and remove them from dir_subs
    '''
    nested = set()
    for b, box in box_dict.items():
        for c in box.conds:
            if c[0] in ['IMP', 'DUP', 'DIS']:
                r, b1, b2 = c
                dir_subs.remove((b,b1))
                # dir_subs.discard((b,b2)) # this is not necessary as it is derivable by transitivity
                nested.update([b1,b2])
                if r == 'IMP':
                    dir_subs.remove((b1,b2))
    return nested

################################
def eliminate_negation(box_dict, vib, dir_subs, disc_rels):
    '''Eliminate negation relation and make it in-built
    '''
    negated = set()
    for r, b1, b2 in disc_rels.copy():
        if r == 'NEGATION':
            negated.add(b2)
            disc_rels.remove((r, b1, b2))
            if all_disc_parents(b2, disc_rels):
                raise RuntimeError("Negated box {} has multiple discourse parents".format(b2))
            dir_subs.remove((b1, b2))
            box_dict[b1].conds.add((r, b2))
            box_dict[b1].subs.add(b2)
            vib.remove(b2)
            # all parents of negated box will be parent of the box containing negation
            for p in all_parents(b2, dir_subs):
                dir_subs.add((p, b1))

################################
def eliminate_implication(box_dict, vib, dir_subs, disc_rels):
    '''Eliminate implication and make it built-in
    '''
    for b1, b2, b3 in get_implication(disc_rels.copy()):
        disc_rels.remove(('CONDITION', b1, b2))
        disc_rels.remove(('CONSEQUENCE', b2, b3))
        dir_subs.remove((b1, b2))
        dir_subs.remove((b2, b3))
        box_dict[b1].conds.add(('IMP', b2, b3))
        box_dict[b1].subs.update([b2, b3])
        vib.remove(b2)
        vib.remove(b3)
        # any parent of imp boxes is a parent of the outer box
        for p in (all_parents(b2, dir_subs) | all_parents(b3, dir_subs)):
            dir_subs.add((p, b1))

################################
def eliminate_disjunction(box_dict, vib, dir_subs, disc_rels):
    '''Eliminate disjunction and make it built-in
    '''
    for b1, b2, b3 in get_disjunction(disc_rels.copy()):
        disc_rels.remove(('ALTERNATION', b1, b2))
        disc_rels.remove(('ALTERNATION', b1, b3))
        dir_subs.discard((b1, b2)) # this might not be a dir_sub
        dir_subs.discard((b1, b3)) # this might not be a dir_sub
        box_dict[b1].conds.add(('DIS', b2, b3))
        box_dict[b1].subs.update([b2, b3])
        vib.remove(b2)
        vib.remove(b3)
        # any parent of imp boxes is a parent of the outer box
        for p in (all_parents(b2, dir_subs) | all_parents(b3, dir_subs)):
            dir_subs.add((p, b1))

###############################################################################
#                   Operations of subordinate relations
###############################################################################

################################
def has_intermediate(x, y, subs):
    'If there is an intermediate element between x and y'
    for a, b in subs:
        if x == a:
            for (c, d) in subs:
                if b == c and y == d:
                    return b

################################
def all_parents(b, subs):
    'A set of all parents of b based on the subs relation'
    return set([ p for (p, c) in subs if b == c ])

################################
def all_disc_parents(b, disc_rel):
    'A set of all parents of b based on the discourse relation'
    return set([ p for r, p, c in disc_rel if b == c ])

################################
def dir_sub(sub_rel):
    '''Keep only direct relations
    '''
    dir_sub = sub_rel.copy()
    temp_sub = set([])
    while dir_sub != temp_sub:
        temp_sub = dir_sub.copy()
        for (x, y) in temp_sub.copy():
            if has_intermediate(x, y, temp_sub):
                dir_sub.remove((x, y))
    return dir_sub

################################
def get_implication(disc_rels):
    for (d, d1, d2) in disc_rels:
        if d == 'CONDITION':
            for (s, s2, s3) in disc_rels:
                if s == 'CONSEQUENCE' and s2 == d2:
                    yield d1, d2, s3

################################
def get_disjunction(disc_rels):
    memory = []
    disj = []
    for (r, d, d1) in disc_rels:
        if r == 'ALTERNATION':
            for (t, a, d2) in disc_rels:
                if t == 'ALTERNATION' and d == a and d2 != d1 and set([d1, d2]) not in memory:
                    memory.append(set([d1, d2]))
                    disj.append((d, d1, d2))
    return disj

################################
def has_dis_rel(rels, disc_rels):
    '''Check if one of the relations is in the discourse relations'''
    for (d, _, _) in disc_rels:
        if d in rels:
            return True

###############################################################################
#                   Operations of box conditions
###############################################################################

################################
def is_tense_cond(cond):
    'Is tense-related condition'
    if cond[0:2] == ('time', '"n.08"'):
        return True
    elif cond[0] == 'Time':
        return True
    elif len(cond) == 3 and cond[2] == '"now"':
        return True
    else:
        return False

################################
def remained_cond_refs(conditions, previous_refs):
    '''Get a set of referents of the conditions that remained after deleting some conditions
    '''
    refs = set()
    for c in conditions:
        if c[0] in ['IMP', 'DUP', 'DIS']:
            continue # ignore as it is not about entities
        for i in c[1:]:
            if i in previous_refs:
                refs.add(i)
    del_cond_refs = previous_refs - refs
    return refs, del_cond_refs

################################
def remove_tense(box_dict, sub_rel):
    '''Remove tense information from DRSs
    '''
    new_box_dict = dict()
    for b, box in box_dict.items():
        c_num = len(box.conds)
        box.conds = set([ c for c in box.conds if not is_tense_cond(c) ])
        if c_num != len(box.conds):
            box.cond_refs, del_cond_refs = remained_cond_refs(box.conds, box.cond_refs)
            # keep those refs that occur in the current conditions
            box.refs = box.refs - del_cond_refs
        # keep even empty boxes, e.g. for negation
        new_box_dict[b] = box
    # if box is removed, remove it from relation
    sub_rel = set([ (b1,b2) for (b1,b2) in sub_rel if b1 in new_box_dict and b2 in new_box_dict ])
    return new_box_dict, sub_rel
