**NLI2FOLI reduces Natural Language Inference (NLI) to First-order Logic Inference (FOLI).**

NLI2FOLI interprets each sentence in an NLI problem as a first-order logic (FOL) formula 
and employs a FOL theorem proving (with [Prover9](https://www.cs.unm.edu/~mccune/prover9/)) 
to reason with the formulas.
The system uses Discourse Representation Structures (DRSs) from 
the [Parallel Meaning Bank](https://pmb.let.rug.nl/) (PMB) 
to model sentence meaning with a FOL formula.

NLI2FOLI was developed to account for NLI problems 
in the [SICK](http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf) dataset. 

NLI2FOLI reads the NLI problems from a certain SICK part.
For each problem, it looks up the corresponding PMB documents based on the information available in `sick2pd.json`.
Formulas for the premise-hypothesis pairs are obtained by reading the `.drs.clf` files, 
ignoring tense-related information, recovering DRSs, and translating the DRSs into FOL formulas.
After the formulas are obtained they are passed to the theorem prover Prover9.
The implementation employs [wrappers and tools from NLTK](https://www.nltk.org/book/ch10.html), 
mainly following [Blackburn & Bos (2005)](http://www.let.rug.nl/bos/comsem/)

In addition to the FOL formulas, NLI2FOLI employs [WordNet](https://wordnet.princeton.edu/) 
to extract relevant hypernymy relations, like `man.n.01` is `person.n.01`, that are used as lexical knowledge.
Lexical knowledge is encoded in FOL formulas, called axioms.
Axioms can be manually added in `knowledge.json`.

## Usage
Read help on how to run NLI2FOLI:
```
$ python3 nli2foli.py  --help
```

Run NLI2FOLI for the trial part of SICK, with verbosity 1, and write predictions in a file (`pmb_SICK` is extracted from `pmb_SICK.zip`):
```
$ python3 nli2foli.py  --pmb pmb_SICK/  --sick SICK/SICK_trial.txt  --sick2pd sick2pd.json  --out predictions/trial.ans  -v 1
```

It is possible to run NLI2FOLI for the problems with certain IDs and manually encode knowledge axioms for the problems:
```
$ python3 nli2foli.py  --pmb pmb_SICK/  --sick SICK/SICK_train.txt  --sick2pd sick2pd.json  --kb knowledge.json  --pids 953 1909
4500 SICK problems read from SICK/SICK_train.txt
PMB part/documents read for 4500 problems

SICK-953 [ENTAILMENT]
	Prem p65/d0060: Two toddlers are eating corndogs in a wagon, which is really small
	Hypo p06/d0061: Two young children are eating corndogs
Extracted axioms: [<AllExpression all x.(toddler_n01(x) -> (child_n01(x) & exists s.(Attribute(x,s) & young_a01(s))))>, <AllExpression all x.(toddler_n01(x) -> child_n01(x))>]
Premise Formula: exists e1 s1 s2 x1 x2 x3.(Attribute(x3,s1) & really_r01(s2) & Location(e1,x3) & toddler_n01(x1) & small_a01(s1) & Patient(e1,x2) & Quantity(x1,2) & Degree(s1,s2) & corndog_n01(x2) & Agent(e1,x1) & wagon_n01(x3) & eat_v01(e1))
Hypothesis Formula: exists e1 s1 x1 x2.(child_n01(x1) & Attribute(x1,s1) & young_a01(s1) & Patient(e1,x2) & Quantity(x1,2) & corndog_n01(x2) & Agent(e1,x1) & eat_v01(e1))
Result for  953: ENTAILMENT vs entailment (Definite answer) Eureka!!!

SICK-1909 [ENTAILMENT]
	Prem p43/d0049: The windows are being polished by a man
	Hypo p42/d0049: The windows are being cleaned by a man
Extracted axioms: [<AllExpression all x.(polish_v01(x) -> clean_v01(x))>]
Premise Formula: exists e1 x1 x2.(polish_v01(e1) & man_n01(x2) & Patient(e1,x1) & Agent(e1,x2) & window_n01(x1))
Hypothesis Formula: exists e1 x1 x2.(Agent(e1,x2) & clean_v01(e1) & Patient(e1,x1) & man_n01(x2) & window_n01(x1))
Result for 1909: ENTAILMENT vs entailment (Definite answer) Eureka!!!
========================================= Status counts (2)=========================================
Definite answer	2
  | E |
--+---+
E |<2>|
--+---+
(row = reference; col = test)
```

If you want to see DRSs, before they are translated into FOL, for particular problems:
```
$ python3 nli2foli.py --pmb pmb_SICK/ --sick SICK/SICK_train.txt --sick2pd sick2pd.json --pids 129 --draw-DRS
```

For evaluation, run the following, which allows to print problems with a certain combination of gold and predicted labels:
```
$ python3 sick_eval.py  --sick SICK/SICK_trial.txt  --pred predictions/trial.ans  --filter CE
```
