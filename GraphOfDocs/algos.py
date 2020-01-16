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
