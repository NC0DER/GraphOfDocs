import sys
import platform
from neo4j import ServiceUnavailable
from GraphOfDocs.neo4j_wrapper import Neo4jDatabase
from GraphOfDocs.utils import generate_words, read_dataset, clear_screen
from GraphOfDocs.create import *

def graphofdocs(create = False):
    uri = 'bolt://localhost:7687'
    username = 'neo4j'
    password = '123'
    current_system = platform.system()
    # Open the database.
    try:
        database = Neo4jDatabase(uri, username, password)
        # Neo4j server is unavailable.
        # This client app cannot open a connection.
    except ServiceUnavailable as error:
        print('\t* Neo4j database is unavailable.')
        print('\t* Please check the database connection before running this app.')
        input('\t* Press any key to exit the app...')
        sys.exit(1)

    if create:
        # Delete nodes from previous iterations.
        database.execute('MATCH (n) DETACH DELETE n', 'w')

        # Create uniqueness constraint on key to avoid duplicate word nodes.
        database.execute('CREATE CONSTRAINT ON (word:Word) ASSERT word.key IS UNIQUE', 'w')

        # Read text from files, which becomes a string in a list called dataset.
        dataset = read_dataset('C:\\Users\\USER\\source\\repos\\GraphOfDocs\\GraphOfDocs\\GraphOfDocs\\20news-18828-all\\')
        count = 1
        total_count = len(dataset)
        # Iterate all file records of the dataset.
        for filename, file in dataset:
            # Print the number of the currently processed file.
            print('Processing ' + str(count) + ' out of ' + str(total_count) + ' datasets...' )
            # Generate the terms from the text of each file.
            words = generate_words(file)
            # Create the graph of words in the database.
            create_graph_of_words(words, database, filename)
            # Update the progress counter.
            count = count + 1
            # Clear the screen to output the update the progress counter.
            clear_screen(current_system)
        run_initial_algorithms(database)
        create_similarity_graph(database)
        create_clustering_tags(database)
    database.close()
    return

if __name__ == "__main__": graphofdocs()
