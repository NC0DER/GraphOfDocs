# Initialize an empty set of edges.
edges = {}
# Initialize an empty list of unique terms.
# We are using a list to preserver order of appearance.
nodes = []
# Globally Increasing id to distinguish between different graph of words, inside the database.
label_id = 1 

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
