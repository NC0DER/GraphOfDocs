"""
This script contains functions that 
select data from the Neo4j database.
"""
def get_similarity_score(database, filename1, filename2):
    query = ('MATCH (d1:Document {filename: "'+ filename1 +'"})'
    '-[r:is_similar]-(d2:Document {filename: "'+ filename2 +'"}) '
    'RETURN r.score')
    results = database.execute(' '.join(query.split()), 'r')
    if (results): #If list not empty, the files are similar
        return results[0][0]
    else:
        return 0.0

def get_document_communities(database):
    query = ('MATCH (d:Document) RETURN d.community, '
            'collect(d.filename) AS files, '
            'count(d.filename) AS file_count '
            'ORDER BY file_count DESC')
    results = database.execute(' '.join(query.split()), 'r')
    return results

def get_document_terms(database, filename, group_by_word_community_id = False):
    if group_by_word_community_id:
        query = ('MATCH (d:Document {filename: "'+ filename +'"})'
                '-[r:includes]->(w:Word) WITH w.community AS community, '
                'w.key AS word, w.pagerank AS pagerank ORDER BY pagerank DESC '
                'RETURN community, collect([word, pagerank]) AS words, '
                'count(word) AS word_count ORDER BY word_count DESC')
    else:
        query = ('MATCH (d:Document {filename: "'+ filename +'"})'
                '-[r:includes]->(w:Word) RETURN w.key, w.pagerank, w.community')
    results = database.execute(' '.join(query.split()), 'r')
    return results

def get_communities_by_tag(database, tag):
    query = ('MATCH (d:Document)-[r:has_tag]->'
            '(w:Word {key: "'+ tag +'"}) '
            'RETURN d.community, collect(d.filename) AS files')
    results = database.execute(' '.join(query.split()), 'r')
    return results

def get_communities_by_tags(database, tags):
    query = ('MATCH (d:Document)-[r:has_tag]->'
            '(w:Word) WHERE w.key in ' + str(tags) +
            'RETURN d.community, collect(d.filename) AS files')
    results = database.execute(' '.join(query.split()), 'r')
    return results

def get_word_digrams_by_filename(database, filename):
    query = ('MATCH (d:Document {filename: "'+ filename +'"})'
            '-[:includes]->(w1:Word)-[r:connects]->(w2:Word)'
            '<-[:includes]-(d) WHERE id(w1) < id(w2) '
            'WITH w1.key AS source, w2.key AS target, r.weight AS weight '
            'ORDER BY weight DESC RETURN collect([source, target, weight]) AS digrams')
    results = database.execute(' '.join(query.split()), 'r')
    return results