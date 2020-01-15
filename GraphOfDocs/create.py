import platform
from GraphOfDocs.utils import clear_screen

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

    # We are getting the unique terms for the current graph of words.
    terms = []
    for word in words:
        if word not in terms: 
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

def create_similarity_graph(database):
    """
    Function that creates a similarity graph
    based on Jaccard similarity measure.
    This measure connects the document nodes with each other
    using the relationship 'is_similar', 
    which has the similarity score as a property.
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
    'CALL algo.similarity.jaccard(Data, {topK: 1, similarityCutoff: 0.23, write: true, writeRelationshipType: "is_similar", writeProperty: "score"}) '
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

def generate_community_tags_scores(database, community):
    """
    This function generates the most important terms that describe
    a community of similar documents, alongside their pagerank and in-degree scores.
    """
    # Get all intersecting nodes of the speficied community, 
    # ranked by their in-degree (which shows to how many documents they belong to).
    # and pagerank score in descending order.
    query = ('MATCH p=((d:Document {community: '+ str(community) +'})-[:includes]->(w:Word)) '
             'WITH w, count(p) as degree '
             'WHERE degree > 1 '
             'RETURN w.key, w.pagerank as pagerank, degree '
             'ORDER BY degree DESC, pagerank DESC')
    tags_scores = database.execute(' '.join(query.split()), 'r')
    return tags_scores

def create_clustering_tags(database, top_terms = 25):
    """
    This functions creates, in the Neo4j database, 
    for all communities, the relationships that connect 
    document nodes of a similarity community with top important 
    clustering tags for that community, based on the amount of common
    appearances between documents and a higher pagerank score.
    """
    current_system = platform.system()
    # Remove has_tag edges from previous iterations.
    database.execute('MATCH ()-[r:has_tag]->() DELETE r', 'w')
    # Get all id numbers from communities and all their assosiated file(name)s.
    print('Loading all community ids and their filenames...')
    query = ('MATCH (d:Document) RETURN d.community, '
            'collect(d.filename) AS files, '
            'count(d.filename) AS file_count '
            'ORDER BY file_count DESC')
    results = database.execute(' '.join(query.split()), 'r')
    # The communities are ordered by filecount, which means that after the first one found,
    # with 1 file all the rest have the same amount of documents.
    # These communities are a side effect of the Louvain implementation of Neo4j.
    # There is no reason to create tags in isolated communities, since there are no common tags, 
    # with other documents. Therefore we are going to filter them out of the results list.
    index = 0
    for result in results:
        if result[2] == 1: # filecount == 1
            break
        index = index + 1
    # Slice the list based on the first found index.
    results = results[:index]
    # Count all results (rows) for a simple loading screen.
    count = 1
    total_count = len(results)
    for [community, filenames, filecount] in results:
        # Print the number of the currently processed community.
        print('Processing ' + str(count) + ' out of ' + str(total_count) + ' communities...' )
        tags_scores = generate_community_tags_scores(database, community)
        # Get the top 25 tags from the tags and scores list.
        top_tags = [tag[0] for tag in tags_scores[:top_terms]]
        for filename in filenames:
            # Connect a filename of a specific community with all its associated tags,
            # where tags are important words that describe that particular community,
            # and which already exist in the graph of docs model.
            query = ('MATCH (d:Document {filename: "'+ filename +'"}) '
                     'UNWIND ' + str(top_tags) +' AS tag '
                     'MATCH (w:Word {key: tag}) '
                     'CREATE (d)-[r:has_tag]->(w) ')
            database.execute(' '.join(query.split()), 'w')
        # Update the progress counter.
        count = count + 1
        # Clear the screen to output the update the progress counter.
        clear_screen(current_system)
    return
