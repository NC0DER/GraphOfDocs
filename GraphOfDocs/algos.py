"""
This script contains wrapper functions that 
call algorithms in the database,
such as Pagerank, Louvain Community Detection,
and Jaccard Similarity Measure.
Their implementantions are located
in the Neo4j Algorithms library.
"""

def pagerank(node, edge, iterations, property):
    query = ('CALL algo.pageRank("'+ node +'", "'+ edge +'", '
            '{iterations: '+ iterations +', dampingFactor: 0.85, write: true, writeProperty: "'+ property +'"}) '
            'YIELD nodes, iterations, loadMillis, computeMillis, writeMillis, dampingFactor, write, writeProperty')
    database.execute(' '.join(query.split()), 'w')
    return

def louvain(node, edge, property):
    query = ('CALL algo.louvain("'+ node +'", "'+ edge +'", '
            '{direction: "BOTH", writeProperty: "'+ property +'"}) '
            'YIELD nodes, communityCount, iterations, loadMillis, computeMillis, writeMillis')
    database.execute(' '.join(query.split()), 'w')
    return
def jaccard(source, edge, target, cutoff, relationship, property):
    query = ('MATCH (d:'+ source + ')-[:'+ edge +']->(w:'+ target + ') '
    'WITH {item:id(d), categories: collect(id(w))} as data '
    'WITH collect(data) as Data '
    'CALL algo.similarity.jaccard(Data, {topK: 1, similarityCutoff: '+ cutoff +', write: true, writeRelationshipType: "'+ relationship +'", writeProperty: "'+ property +'"}) '
    'YIELD nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, stdDev, p25, p50, p75, p90, p95, p99, p999, p100 '
    'RETURN nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, p95 ')
    database.execute(' '.join(query.split()), 'w')
    return