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
from neo4j_wrapper import Neo4jDatabase, ServiceUnavailable

lemmatizer = WordNetLemmatizer() # Initialize lemmatizer once.
stemmer = PorterStemmer() # Initialize Porter's stemmer once.

id = 0 # Globally Increasing id to avoid duplicate edges, since the keys of the nodes are duplicated.
label_id = 1 # Globally Increasing id to distinguish between different graph of words, inside the database.

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
    """
    Function that creates a Graph of Words inside the neo4j database,
    using the appropriate cypher queries.
    """
    # Initialize an empty set of edges.
    edges = {}

    # Get a list of unique terms from the list of words.
    terms = []
    for word in words:
        if word not in terms: terms.append(word)

    # Store non-duplicated words in a dictionary,
    # which is 0 initialized, until it gets the ids
    # for each word.
    global id # Using the globally increasing node id.
    global label_id # Using the globally increasing label id.
    dictionary = {word: 0 for word in terms}
    for word, _ in dictionary.items():
        res = database.execute('CREATE (w:Word'+ str(label_id) +' {id: '+ str(id) +', key: "'+ word +'"})', 'w')
        # Assigning an id to each word, after its used, we increase to get the next one.
        dictionary[word] = id
        id = id + 1

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
        # Get the unique id of the current word.
        current_id = dictionary[current]
        # Connect the current element with the next elements of the window size.
        for j in range(1, window_size):
            next = words[i + j]
            next_id = dictionary[next]
            edge = (current, next)
            if edge in edges:
                # If the edge, exists just update its weight.
                edges[edge] += 1
                query = ('MATCH (w1:Word'+ str(label_id) +' {key: "'+ current +'"})-[r:connects]->(w2:Word'+ str(label_id) +' {key: "' + next + '"}) '
                        'WHERE w1.id = ' + str(current_id) + ' AND w2.id = ' + str(next_id) + ' '
                        'SET r.weight = '+ str(edges[edge]))
            else:
                # Else, create it, with a starting weight of 1 meaning first co-occurence.
                edges[edge] = 1
                query = ('MATCH (w1:Word'+ str(label_id) +' {key: "'+ current +'"}) '
                        'MATCH (w2:Word'+ str(label_id) +' {key: "' + next + '"}) '
                        'WHERE w1.id = ' + str(current_id) + ' AND w2.id = ' + str(next_id) + ' '
                        'CREATE (w1)-[r:connects {weight:' + str(edges[edge]) + '}]->(w2) ')
            res = database.execute(' '.join(query.split()), 'w')
    label_id = label_id + 1
    return

def read_datasets(filepath):
    """
    Function that gets a list of filenames in the directory specified by filepath,
    then reading them in text mode, and appending them in a list which is to be returned.
    """
    data = []
    files = [file for file in listdir(filepath) if isfile(join(filepath, file))]
    for file in files:
        with open(''.join([filepath, file]), 'r', encoding = 'utf8') as fd:
            data.append(fd.read().replace('\n', ''))
    return data

def clear_screen(current_system):
    if current_system == 'Windows':
        system('cls')
    else:
        system('clear') # Linux/OS X.
    return

def main():
    uri = 'bolt://localhost:7687'
    username = 'neo4j'
    password = '123'
    current_system = platform.system()
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
    #test = str("A method for solution of systems of linear algebraic equations with m-dimensional lambda matrices. A system of linear algebraic equations with m-dimensional lambda matrices is considered. The proposed method of searching for the solution of this system lies in reducing it to a numerical system of a special kind.")
    #words = generate_words(test)
    #print(words)
    database.execute('MATCH (n) DETACH DELETE n', 'w')
    database.execute('MATCH (n) DETACH DELETE n', 'w')
    datasets = read_datasets('C:\\Users\\USER\\source\\repos\\GraphOfWords\\GraphOfWords\\CV Datasets\\')
    count = 1
    total_count = len(datasets)
    for dataset in datasets:
        print('Processing ' + str(count) + ' out of ' + str(total_count) + ' datasets...' )
        words = generate_words(dataset)
        create_graph_of_words(words, database)
        count = count + 1
        clear_screen(current_system)
    return



if __name__ == "__main__": main()
