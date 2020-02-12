import argparse
"""This script contains code for the command line argument parser."""
parser = argparse.ArgumentParser(description = 'Create, reinitialize, analyze the graphofdocs model')
parser.add_argument('-c', '--create', action = 'store_true', 
                    help = 'set this flag to create and initialize the graphofdocs model')

parser.add_argument('-r', '--reinitialize', action = 'store_true', 
                    help = 'set this flag to reinitialize the graphofdocs model, '
                    'by re-running centrality, community detection and similarity algorithms')

parser.add_argument('-dir', '--dirpath', nargs = 1, type = str,
                    help = 'if create is set, ' 
                    'then specify a directory path, '
                    'containing plaintext files (documents).')

parser.add_argument('-ws', '--window-size', nargs = 1, type = int, 
                    default = [4], choices = [2, 3, 4, 5, 6],
                    help = 'if create is set, then set the window size')

parser.add_argument('-e', '--extend-window', action = 'store_true',
                    help = 'if create is set, then set this flag to '
                    'enable the sliding text window to extend '
                    'over words of different sentences. '
                    '(Default behavior: Disabled)')

parser.add_argument('-is', '--insert-stopwords', action = 'store_true',
                    help = 'if create is set, then set this flag to '
                    'enable the insertion of stopwords from the text to the model. '
                    '(Default behavior: Disabled)')

parser.add_argument('-l', '--lemmatize', action = 'store_true',
                    help = 'if create is set, then set this flag to '
                    'enable the lemmatization of terms of the text. '
                    '(Default behavior: Disabled)')

parser.add_argument('-s', '--stem', action = 'store_true',
                    help = 'if create is set, then set this flag to '
                    'enable the stemming of terms of the text. '
                    '(Default behavior: Disabled)')
