import sys
import platform
from os import system
from os import listdir
from os.path import isfile, join
from string import punctuation
from nltk import pos_tag, sent_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer() # Initialize lemmatizer once.
stemmer = PorterStemmer() # Initialize Porter's stemmer once.

stop_words = set(stopwords.words('english')).union([ # Augment the stopwords set.
    'don','didn', 'doesn', 'aren', 'ain', 'hadn',
    'hasn', 'mightn', 'mustn', 'couldn', 'shouldn',
    've', 'll', 'd', 're', 't', 's'])

def get_wordnet_tag(tag):
    """
    Function that maps default part-of-speech 
    tags to wordnet part-of-speech tags.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else: #default lemmatizer parameter
        return wordnet.NOUN

def generate_words(text_corpus, remove_stopwords = True, lemmatize = False, stemming = False):
    """
    Function that generates words from a text corpus and optionally lemmatizes them.
    Returns a set of unique tokens based on order of appearance in-text.
    """
    # Remove all whitespace characters (by split) and join on space.
    text_corpus = ' '.join(text_corpus.split())
    # Handle special characters that connect words.
    text_corpus = text_corpus.translate({ord(c): '' for c in '\'\"'})
    # Find all end of sentences and introduce a special string to track them.
    # By chaining the replace methods together, we achieve a slight amount of performance,
    # over other methods, that achieve the same result.
    text_corpus = text_corpus.replace('. ', ' e5 ')\
                .replace('! ', ' e5 ' )\
                .replace('? ', ' e5 ' )
    # Leftover characters (e.g from domains) are being translated into spaces.
    text_corpus = text_corpus.replace('@', ' ')\
                .replace('.', ' ')\
                .replace('!', ' ')\
                .replace('?', ' ')
    # Remove punctuation and lowercase the string.
    input_text = text_corpus.translate(str.maketrans(' ', ' ', punctuation)).lower()
    if remove_stopwords:
        tokens = [token for token in word_tokenize(input_text) if not token in stop_words] 
    else:
        tokens = word_tokenize(input_text)
    if lemmatize:
        tokens_tags = pos_tag(tokens) # Create part-of-speech tags.
        # Overwrite the list with the lemmatized versions of tokens.
        tokens = [lemmatizer.lemmatize(token, get_wordnet_tag(tag)) for token, tag in tokens_tags]
    if stemming:
        # Overwrite the list with the stemmed versions of tokens.
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def read_datasets(filepath):
    """
    Function that gets a list of filenames in the directory specified by filepath,
    then reading them in text mode, and appending them in a list which contains the file(name),
    and its contents, which have newline characters removed.
    Handles newline endings of '\n' and '\r\n'.
    """
    data = []
    files = [file for file in listdir(filepath) if isfile(join(filepath, file))]
    for file in files:
        with open(''.join([filepath, file]), 'rt', encoding = 'utf-8-sig') as fd:
            data.append((file, fd.read().replace('\n', ' ').replace('\r', '')))  
    return data

def clear_screen(current_system):
    if current_system == 'Windows':
        system('cls')
    else:
        system('clear') # Linux/OS X.
    return
