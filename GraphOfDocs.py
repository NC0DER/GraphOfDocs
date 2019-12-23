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

# Initialize an empty set of edges.
edges = {}
# Initialize an empty list of unique terms.
# We are using a list to preserver order of appearance.
nodes = []

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

def generate_words(text_corpus, remove_stopwords = True, lemmatize = False, stemming = False):
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

def create_graph_of_words(words, database, filename, window_size = 4):
    """
    Function that creates a Graph of Words that contains all nodes from each document for easy comparison,
    inside the neo4j database, using the appropriate cypher queries.
    """
    # We are using a global set of edges to avoid creating duplicate edges between different graph of words.
    # Basically the co-occurences will be merged.
    global edges

    # We are using a global set of edges to avoid creating duplicate nodes between different graph of words.
    # A list is being used to respect the order of appearance.
    global nodes

    # We are getting the unique terms for the current graph of words,
    # And we are also cleaning the data, from numbers and leftover syllabes or letters.
    terms = []
    for word in words:
        if word not in terms and not word.isnumeric() and len(word) > 2: 
            terms.append(word)

    # Using the globally increasing label id, each document has its own id.
    global label_id 
    for word in terms:
        # If the word is already a node, then simply update its label.
        if word in nodes:
            database.execute('MATCH (w:Word {key: "'+ word +'"}) SET w:Document' + str(label_id), 'w')
        # If not then create it.
        else:
            database.execute('CREATE (w:Word:Document'+ str(label_id) +' {key: "'+ word +'"})', 'w')
            # Append word to the global node graph, to avoid duplicate creation.
            nodes.append(word)

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
            edge = (current, next)
            if edge in edges:
                # If the edge, exists just update its weight.
                edges[edge] += 1
                query = ('MATCH (w1:Word:Document'+ str(label_id) +' {key: "'+ current +'"})-[r:connects]->(w2:Word:Document'+ str(label_id) +' {key: "' + next + '"}) '
                        'SET r.weight = '+ str(edges[edge]))
            else:
                # Else, create it, with a starting weight of 1 meaning first co-occurence.
                edges[edge] = 1
                query = ('MATCH (w1:Word:Document'+ str(label_id) +' {key: "'+ current +'"}) '
                        'MATCH (w2:Word:Document'+ str(label_id) +' {key: "' + next + '"}) '
                        'CREATE (w1)-[r:connects {weight:' + str(edges[edge]) + '}]->(w2) ')
                database.execute(' '.join(query.split()), 'w')

    # Create a parent node that represents the document itself.
    # This node is connected to all words of its own graph,
    # and will be used for similarity/comparison queries.
    database.execute('CREATE (p:Head {key: "Document'+ str(label_id) +'", filename: "'+ filename +'"})', 'w')
    query = ('MATCH (d:Document'+ str(label_id) +') WITH collect(d) as words '
            'MATCH (h:Head {key: "Document'+ str(label_id) +'"}) '
            'UNWIND words as word '
            'CREATE (h)-[:includes]->(word)')
    database.execute(' '.join(query.split()), 'w')
    # All queries are finished so increase the global label id, to process the next graph of words. 
    label_id = label_id + 1
    return

def run_algorithms(database):
    """
    Function that runs centrality & community detection algorithms,
    in order to prepare the data for analysis and visualization.
    Weighted Pagerank & Louvain are used, respectively.
    The calculated score for each node of the algorithms is being stored
    on the nodes themselves.
    """
    query = ('CALL algo.pageRank("Word", "connects", '
            '{iterations: 20, dampingFactor: 0.85, write: true, writeProperty: "pagerank"}) '
            'YIELD nodes, iterations, loadMillis, computeMillis, writeMillis, dampingFactor, write, writeProperty')
    database.execute(' '.join(query.split()), 'w')
    query = ('CALL algo.louvain("Word", "connects", '
            '{direction: "BOTH", writeProperty: "community"}) '
            'YIELD nodes, communityCount, iterations, loadMillis, computeMillis, writeMillis')
    database.execute(' '.join(query.split()), 'w')


def read_datasets(filepath):
    """
    Function that gets a list of filenames in the directory specified by filepath,
    then reading them in text mode, and appending them in a list which contains the file(name) and its contents.
    """
    data = []
    files = [file for file in listdir(filepath) if isfile(join(filepath, file))]
    for file in files:
        with open(''.join([filepath, file]), 'r', encoding = 'utf8') as fd:
            data.append((file, fd.read().replace('\n', '')))  
    return data

def clear_screen(current_system):
    if current_system == 'Windows':
        system('cls')
    else:
        system('clear') # Linux/OS X.
    return

def main(create = True):
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
    if create:
        # Delete nodes from previous iterations.
        database.execute('MATCH (n) DETACH DELETE n', 'w')
        database.execute('MATCH (n) DETACH DELETE n', 'w')

        # Create uniqueness constraint on key to avoid duplicate word nodes.
        database.execute('CREATE CONSTRAINT ON (word:Word) ASSERT word.id IS UNIQUE', 'w')

        # Read text from files, which becomes a string in a list called datasets.
        datasets = read_datasets('C:\\Users\\USER\\source\\repos\\GraphOfWords\\GraphOfWords\\wiki20\\documents\\')
        count = 1
        total_count = len(datasets)
        # Iterate all datasets.
        for filename, dataset in datasets:
            # Print the number of the currently processed dataset.
            print('Processing ' + str(count) + ' out of ' + str(total_count) + ' datasets...' )
            # Generate the terms from the text of each dataset.
            words = generate_words(dataset)
            # Create the graph of words in the database.
            create_graph_of_words(words, database, filename)
            # Update the progress counter.
            count = count + 1
            # Clear the screen to output the update the progress counter.
            clear_screen(current_system)
    run_algorithms(database)
    return

if __name__ == "__main__": main()
