# Computational Semantics course @ University of Groningen
The repository contains scripts that are provided in the [Computational Semantics course](https://www.rug.nl/ocasys/rug/vak/show?code=LIX021M05).

## Comparing tok.off files (`tok_off_compare.py`)

The script compares tok.off files. First the files are converted into TOIS-annotation format (i.e. each character is labeled with `T`oken start, `O`utside a token, `I`n token, or `S`entence start) and then a confusion matrix of these two annotations is built. More details about the TOIS-annotation can be found in [Elephant paper](http://aclweb.org/anthology/D/D13/D13-1146.pdf)

The script `tok_off_compare.py` can be used in two ways: 
1. Compare **two concrete tok.off files** (i.e. files formatted as `en.tok.off`) and get the confusion matrix over `T` `O` `I` `S` labels.
It also expects that a raw file is provided. 
2. Compare tok.off files for every document in the **part directory** (i.e. pXX).
It is expected that in every dXX directory there will be the original `en.tok.off` and your `en.tok.off` file. 
The final confusion matrix is a sum of confusion matrices per document.

### Usage ###

The code supports **verbosity levels**. Higher levels entail lower levels:  
`-v 0`: (by default) only the counter is shown as a progress bar.  
`-v 1`: show the different TOIS-annotations under each other.  
`-v 2`: show the TOIS-annotations for every comparison.  
`-v 3`: report file names that are under comparison.  
`-v 4`: report details about TOIS-annotation recovery per tok.off files.  

Use `--help` for the details of the usage.

### Examples ###
1. Run **comparison for two tok.off files** (with `verbosity=3` and `beam=110`) with a specific `--rawfile`

   ```./tok_off_compare.py -f p12/d1234/en.tok.off  en.tok.off.s1234567  -r p12/d1234/en.raw  -v 3 -b 110```

   The output will be a list of working files, a contrast between two TOIS-annotations and a confusion matrix for two TOIS-annotations:

   ```
   Working with ./p12/d1234/en.raw ./p12/d1234/en.tok.off ./en.tok.off.s1234567
   1LINE_RAW:Around 120 miles of Cornish coast and 80 kilometres of France was contaminated and around 15,000 sea
   TOIS_ORIG:#----- ^-- ^---- ^- ^------ ^---- ^-- ^- ^--------- ^- ^----- ^-- ^----------- ^-- ^----- ^----- ^--
   TOIS_MINE:#----- ^-- ^---- ^- ^------ ^---- ^-- ^- ^--------- ^- ^----- ^-- ^----------- ^-- ^----- ^----- ^--
   SHOW_DIFF:
   1LINE_RAW:birds killed along with huge numbers of marine organisms before the 270 square mile slick dispersed.
   TOIS_ORIG:^---- ^----- ^---- ^--- ^--- ^------ ^- ^----- ^-------- ^----- ^-- ^-- ^---------- ^---- ^--------^
   TOIS_MINE:^---- ^----- ^---- ^--- ^--- ^------ ^- ^----- ^-------- ^----- ^-- ^-- ^-----      ^---- ^--------^
   SHOW_DIFF:                                                                              $$$$$
   ============ Confusion Matrix ============
   ORIG\MINE    T    O    I    S
   ---------------------------------------------
       T       32    0    0    0
       O        0   31    0    0
       I        0    5  132    0
       S        0    0    0    1
   ```


2. Run **comparison for the Part directory**, e.g., `p12`. 
It is expected that in every `p12/dXXXX` directory there are `en.tok.off`, `en.raw` and `en.tok.off.s1234567` files. 
By default `verbosity=0` and only the progress counter will be shown with the final confusion matrix.  

   ```./tok_off_compare.py  -d p12  -m 'en.tok.off.s1234567'``` 
   
   The matrix will look like this and it counts TOIS-annotations for all the documents in the part for which the necessary tok.off files existed:
   ```   
   ============ Confusion Matrix ============
   ORIG\MINE       T       O       I       S
   ---------------------------------------------
       T       43065       0      22       0
       O           0   37592      22       0
       I           0       0  145412       0
       S           0       0       0    3727
   ```
### Notes ###
- The script does not check validity of your tok.off files.
- If you *slightly mess up* token IDs in your tok.off file the script will close eyes on it (a side effect). 
For example, if the tok.off has the fragment below, `101 106 1017 birds` will not cause an error and it will be treated as if there was `101 106 1018 birds`.
   ```
   97 100 1017 sea
   101 106 1017 birds
   107 113 1019 killed
   ```
   Though character offsets and a sentence ID is taken very seriously!

### System requirements ###
```
The provided code is tested on Ubuntu with python 2.7.6.
It should work for any python of version 2.6 or 2.7.
```



