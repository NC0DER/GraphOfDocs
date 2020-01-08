# Initialize an empty set of edges.
edges = {}
# Initialize an empty list of unique terms.
# We are using a list to preserver order of appearance.
nodes = []

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

    for word in terms:
        # If the word doesn't exist as a node, then create it.
        if word not in nodes:
            database.execute('CREATE (w:Word {key: "'+ word +'"})', 'w')
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
        # If the current word is the end of sentence string,
        # we need to skip it, in order to go to the words of the next sentence,
        # without connecting words of different sentences, in the database.
        if current == 'e5':
            continue
        # Connect the current element with the next elements of the window size.
        for j in range(1, window_size):
            next = words[i + j]
            # Reached the end of sentence string.
            # We can't connect words of different sentences,
            # therefore we need to pick a new current word,
            # by going back out to the outer loop.
            if next == 'e5':
                break
            edge = (current, next)
            if edge in edges:
                # If the edge, exists just update its weight.
                edges[edge] = edges[edge] + 1
                query = ('MATCH (w1:Word {key: "'+ current +'"})-[r:connects]-(w2:Word {key: "' + next + '"}) '
                        'SET r.weight = '+ str(edges[edge]))
            else:
                # Else, create it, with a starting weight of 1 meaning first co-occurence.
                edges[edge] = 1
                query = ('MATCH (w1:Word {key: "'+ current +'"}) '
                        'MATCH (w2:Word {key: "' + next + '"}) '
                        'MERGE (w1)-[r:connects {weight:' + str(edges[edge]) + '}]-(w2) ')
            # This line of code, is meant to be executed, in both cases of the if...else statement.
            database.execute(' '.join(query.split()), 'w')

    # Create a parent node that represents the document itself.
    # This node is connected to all words of its own graph,
    # and will be used for similarity/comparison queries.
    database.execute('CREATE (d:Document {filename: "'+ filename +'"})', 'w')
    # Create a word list with comma separated, quoted strings for use in the Cypher query below.
    word_list = ', '.join('"{0}"'.format(word) for word in set(words))
    query = ('MATCH (w:Word) WHERE w.key IN [' + word_list + '] '
            'WITH collect(w) as words '
            'MATCH (d:Document {filename: "'+ filename +'"}) '
            'UNWIND words as word '
            'CREATE (d)-[:includes]->(word)')
    database.execute(' '.join(query.split()), 'w')
    return

def run_initial_algorithms(database):
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

def create_similarity_graph(database, system):
    """
    Function that creates a similarity graph
    based on Jaccard similarity measure.
    This measure connects the document nodes with each other
    using the relationship 'is_similar', which has the similarity score as a property.
    In order to prepare the data for analysis and visualization,
    we use Louvain Community detection algorithm.
    The calculated community id for each node is being stored
    on the nodes themselves.
    """
    # Remove similarity edges from previous iterations.
    database.execute('MATCH ()-[r:is_similar]->() DELETE r', 'w')

    # Create the similarity graph using Jaccard similarity measure.
    query = ('MATCH (d:Document)-[:includes]->(w:Word) '
    'WITH {item:id(d), categories: collect(id(w))} as data '
    'WITH collect(data) as Data '
    'CALL algo.similarity.jaccard(Data, {topK: 1, similarityCutoff: 0.2, write: true, writeRelationshipType: "is_similar", writeProperty: "score"}) '
    'YIELD nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, stdDev, p25, p50, p75, p90, p95, p99, p999, p100 '
    'RETURN nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, p95 ')
    database.execute(' '.join(query.split()), 'w')

    # Find all similar document communities.
    query = ('CALL algo.louvain("Document", "is_similar", '
            '{direction: "BOTH", writeProperty: "community"}) '
            'YIELD nodes, communityCount, iterations, loadMillis, computeMillis, writeMillis')
    database.execute(' '.join(query.split()), 'w')
    print('Similarity graph created.')
    return
