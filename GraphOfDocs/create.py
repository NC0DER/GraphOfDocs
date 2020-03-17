"""
This script contains functions that 
create data in the Neo4j database.
"""
import platform
from GraphOfDocs.utils import clear_screen
from GraphOfDocs.algos import *
from GraphOfDocs.select import get_communities_filenames, get_communities_tags

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

    # Files that have word length < window size, are skipped.
    # Window size ranges from 2 to 6.
    length = len(words)
    if (length < window_size):
        # Early exit, we return the skipped filename
        return filename

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
    # Remove end-of-sentence token, so it doesn't get created.
    if 'e5c' in terms:
        terms.remove('e5c')
    # If the word doesn't exist as a node, then create it.
    for word in terms:
        if word not in nodes:
            database.execute(f'CREATE (w:Word {{key: "{word}"}})', 'w')
            # Append word to the global node graph, to avoid duplicate creation.
            nodes.append(word)      

    

    # Create unique connections between existing nodes of the graph.
    for i, current in enumerate(words):
        # If there are leftover items smaller than the window size, reduce it.
        if i + window_size > length:
            window_size = window_size - 1
        # If the current word is the end of sentence string,
        # we need to skip it, in order to go to the words of the next sentence,
        # without connecting words of different sentences, in the database.
        if current == 'e5c':
            continue
        # Connect the current element with the next elements of the window size.
        for j in range(1, window_size):
            next = words[i + j]
            # Reached the end of sentence string.
            # We can't connect words of different sentences,
            # therefore we need to pick a new current word,
            # by going back out to the outer loop.
            if next == 'e5c':
                break
            edge = (current, next)
            if edge in edges:
                # If the edge, exists just update its weight.
                edges[edge] = edges[edge] + 1
                query = (f'MATCH (w1:Word {{key: "{current}"}})-[r:connects]-(w2:Word {{key: "{next}"}}) '
                         f'SET r.weight = {edges[edge]}')
            else:
                # Else, create it, with a starting weight of 1 meaning first co-occurence.
                edges[edge] = 1
                query = (f'MATCH (w1:Word {{key: "{current}"}}) '
                         f'MATCH (w2:Word {{key: "{next}"}}) '
                         f'MERGE (w1)-[r:connects {{weight: {edges[edge]}}}]-(w2)')
            # This line of code, is meant to be executed, in both cases of the if...else statement.
            database.execute(query, 'w')

    # Create a parent node that represents the document itself.
    # This node is connected to all words of its own graph,
    # and will be used for similarity/comparison queries.
    database.execute(f'CREATE (d:Document {{filename: "{filename}"}})', 'w')
    # Create a word list with comma separated, quoted strings for use in the Cypher query below.
    #word_list = ', '.join(f'"{word}"' for word in terms)
    query = (f'MATCH (w:Word) WHERE w.key IN {terms} '
              'WITH collect(w) as words '
             f'MATCH (d:Document {{filename: "{filename}"}}) '
              'UNWIND words as word '
              'CREATE (d)-[:includes]->(word)')
    database.execute(query, 'w')
    return

def run_initial_algorithms(database):
    """
    Function that runs centrality & community detection algorithms,
    in order to prepare the data for analysis and visualization.
    Pagerank & Louvain are used, respectively.
    The calculated score for each node of the algorithms is being stored
    on the nodes themselves.
    """
    # Append the parameter 'weight' for the weighted version of the algorithm.
    pagerank(database, 'Word', 'connects', 20, 'pagerank')
    louvain(database, 'Word', 'connects', 'community')
    return

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
    jaccard(database, 'Document', 'includes', 'Word', 0.23, 'is_similar', 'score')

    # Find all similar document communities.
    # Append the parameter 'score' for the weighted version of the algorithm.
    louvain(database, 'Document', 'is_similar', 'community')
    print('Similarity graph created.')
    return

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
    results = get_communities_filenames(database)

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

    # Get all top tags for each community.
    top_tags = get_communities_tags(database, top_terms)

    for [community, filenames, _] in results:
        # Print the number of the currently processed community.
        print(f'Processing {count} out of {total_count} communities...' )
        try:
            tags = top_tags[community]
        except KeyError:
            print('\t* Error: Community key should exist in dictionary!')

        # Connect filenames of a specific community with all their associated tags.
        # Tags are considered to be important words that describe that community,
        # and which already exist in the graphofdocs model.
        query = (f'UNWIND {filenames} AS filename '
                  'MATCH (d:Document {filename: filename}) '
                 f'UNWIND {tags} AS tag '
                  'MATCH (w:Word {key: tag}) '
                  'CREATE (d)-[r:has_tag]->(w)')
        database.execute(query, 'w')

        # Update the progress counter.
        count = count + 1
        # Clear the screen to output the update the progress counter.
        clear_screen(current_system)
    return
