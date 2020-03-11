"""
This script contains functions that 
select data from the Neo4j database.
"""

def get_communities_filenames(database):
    """
    This function retrieves all filenames (and the file count) 
    for every community of similar documents.
    """
    query = ('MATCH (d:Document) RETURN d.community, '
             'collect(d.filename) AS files, '
             'count(d.filename) AS file_count '
             'ORDER BY file_count DESC')
    results = database.execute(' '.join(query.split()), 'r')
    return results

def get_communities_tags(database, top_terms = None):
    """
    This function generates the most important terms that describe
    each community of similar documents, and returns them for all communities.
    """
    # Get all intersecting nodes of the speficied community, 
    # ranked by their in-degree (which shows to how many documents they belong to).
    # and pagerank score in descending order.
    top_tags = {}
    query = ('MATCH p=((d:Document)-[:includes]->(w:Word)) '
            'WITH d.community as community, w, count(p) as degree '
            'WHERE degree > 1 '
            'WITH community as com, w.key as word, w.pagerank as pagerank, degree as deg '
            'ORDER BY com, deg DESC, pagerank DESC '
            'RETURN com, collect([word, pagerank, deg]) ')
    communities = database.execute(' '.join(query.split()), 'r')

    # Get the top tags from the tags and scores list.
    for [community, tags_scores] in communities:
        # Get all top terms for this community.
        if top_terms is None: 
            top_tags[community] = [tag[0] for tag in tags_scores]
        else:
            top_tags[community] = [tag[0] for tag in tags_scores[:top_terms]]
    return top_tags

def get_word_digrams_by_filename(database, filename):
    query = ('MATCH (d:Document {filename: "'+ filename +'"})'
            '-[:includes]->(w1:Word)-[r:connects]->(w2:Word)'
            '<-[:includes]-(d) WHERE id(w1) < id(w2) '
            'WITH w1.key AS source, w2.key AS target, r.weight AS weight '
            'ORDER BY weight DESC RETURN collect([source, target, weight]) AS digrams')
    results = database.execute(' '.join(query.split()), 'r')
    return results
