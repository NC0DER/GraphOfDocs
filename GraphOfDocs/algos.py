"""
This script contains wrapper functions that 
call algorithms in the database,
such as Pagerank, Louvain Community Detection,
and Jaccard Similarity Measure.
Their implementantions are located
in the Neo4j Algorithms library.
"""

def pagerank(database, node, edge, iterations, property, weight = ''):
    type_correct = all([isinstance(node, str),
    isinstance(edge, str),
    isinstance(iterations, int),
    isinstance(property, str),
    isinstance(weight, str)])
    
    if not type_correct:
        raise TypeError('All arguments should be strings, except iterations which should be int!')

    if weight: # If weight is not an empty str.
        weight = f', weightProperty: {weight}'

    query = (f'CALL algo.pageRank("{node}", "{edge}", '
             f'{{iterations: {iterations}, dampingFactor: 0.85, write: true, writeProperty: "{property}"'+ weight +'}) '
              'YIELD nodes, iterations, loadMillis, computeMillis, writeMillis, dampingFactor, write, writeProperty')
    database.execute(query, 'w')
    return

def louvain(database, node, edge, property, weight = ''):
    type_correct = all([isinstance(node, str),
    isinstance(edge, str),
    isinstance(property, str),
    isinstance(weight, str)])

    if not type_correct:
        raise TypeError('All arguments should be strings!')

    if weight: # If weight is not an empty str.
        weight = ', weightProperty: "'+ weight +'"'

    query = (f'CALL algo.louvain("{node}", "{edge}", '
             f'{{direction: "BOTH", writeProperty: "{property}"'+ weight +'}) '
             'YIELD nodes, communityCount, iterations, loadMillis, computeMillis, writeMillis')
    database.execute(query, 'w')
    return

def jaccard(database, source, edge, target, cutoff, relationship, property):
    type_correct = all([isinstance(source, str),
    isinstance(edge, str),
    isinstance(target, str),
    isinstance(relationship, str),
    isinstance(property, str), 
    isinstance(cutoff, float)])

    if not type_correct:
        raise TypeError('All arguments should be strings, except cutoff which should be a float!')

    query = (
        f'MATCH (d:{source})-[:{edge}]->(w:{target}) '
         'WITH {item:id(d), categories: collect(id(w))} as data '
         'WITH collect(data) as Data '
        f'CALL algo.similarity.jaccard(Data, {{topK: 1, similarityCutoff: {cutoff}, write: true, writeRelationshipType: "{relationship}", writeProperty: "{property}"}}) '
         'YIELD nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, stdDev, p25, p50, p75, p90, p95, p99, p999, p100 '
         'RETURN nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, p95 ')
    database.execute(query, 'w')
    return