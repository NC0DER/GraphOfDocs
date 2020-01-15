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

def get_document_terms(database, filename):
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

