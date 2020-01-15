def get_similarity_score(database, filename1, filename2):
    query = ('MATCH (d1:Document {filename: "'+ filename1 +'"})'
    '-[r:is_similar]-(d2:Document {filename: "'+ filename2 +'"}) '
    'RETURN r.score')
    results = database.execute(' '.join(query.split()), 'r')
    if (results): #If list not empty, the files are similar (> 0.23)
        return results[0][0] #[0.234324234]
    else:
        return 0.0

def get_document_communities(database):
    query = ('MATCH (d:Document) RETURN d.community, '
    'collect(d.filename) AS files, '
    'count(d.filename) AS file_count '
    'ORDER BY file_count DESC')
    results = database.execute(' '.join(query.split()), 'r')
    return results
