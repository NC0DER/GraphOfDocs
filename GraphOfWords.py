import sys
from string import punctuation
from nltk import pos_tag, sent_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from neo4j_wrapper import Neo4jDatabase, ServiceUnavailable

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

def generate_words(text_corpus, remove_stopwords = True, lemmatize = False, stemming = True):
    """
    Function that generates words from a text corpus and optionally lemmatizes them.
    Returns a set of unique tokens based on order of appearance in-text.
    """
    # Handle special characters that connect words.
    text_corpus = text_corpus.translate({ord(c): ' ' for c in '\'\"\\'})
    # Remove punctuation and lowercase the string.
    input_text = text_corpus.translate(str.maketrans('', '', punctuation)).lower()
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

def create_graph_of_words(words, database, window_size = 4):
    # Create non-duplicate words as words of a graph.
    for word in set(words):
        res = database.execute('CREATE (w:Word {key: "'+ word +'"})', 'w')

    # Length should be greater than the window size at all times.
    # Window size ranges from 2 to 6.
    length = len(words)
    try:
        if (length < window_size):
            raise ValueError('Word length should always be bigger than the window size!')
    except ValueError as err:
            print(repr(err))

    # Create unique connections between existing nodes of the graph.
    for i, current in enumerate(words):
        # If there are leftover items smaller than the window size, reduce it.
        if i + window_size > length:
            window_size = window_size - 1
        # Connect the current element with the next elements of the window size.
        for j in range(1, window_size):
            next = words[i + j]
            query = ('MATCH (w1:Word {key: "'+ current +'"}) '
                    'MATCH (w2:Word {key: "' + next + '"}) '
                    'CREATE (w1)-[r:connects]->(w2) ')
            res = database.execute(' '.join(query.split()), 'w')
    return 

def main():
    uri = 'bolt://localhost:7687'
    username = 'neo4j'
    password = '123'
    # Open the database.
    try:
        database = Neo4jDatabase(uri, username, password)
        # Neo4j server is unavailable.
        # This client app cannot open a connection.
    except ServiceUnavailable as error:
        print('\t* Neo4j database is unavailable.')
        print('\t* Please check the database connection before running this app.')
        input('\t* Press any key to exit the app...')
        sys.exit(1)
    test = str("As a discipline, computer science spans a range of topics "
               "from theoretical studies of algorithms and the limits of "
               "computation to the practical issues of implementing "
               "computing systems in hardware and software. ")
    words = generate_words(test)
    print(words)
    # Cleanup all nodes from previous iterations.
    database.execute('MATCH (n) DETACH DELETE n', 'w')

    # Create uniqueness constraint on id / key to avoid duplicate word nodes.
    database.execute('CREATE CONSTRAINT ON (word:Word) ASSERT word.key IS UNIQUE', 'w')
    create_graph_of_words(words, database)

if __name__ == "__main__": main()
