import sys
import platform
from neo4j import ServiceUnavailable
from GraphOfDocs.neo4j_wrapper import Neo4jDatabase
from GraphOfDocs.utils import generate_words, read_dataset, clear_screen
from GraphOfDocs.parse_args import parser
from GraphOfDocs.create import *

def graphofdocs(create, initialize, dirpath, window_size, 
    extend_window, remove_stopwords, lemmatize, stem):

    # List that retains the skipped filenames.
    skipped = []
    current_system = platform.system()
    # Open the database.
    try:
        database = Neo4jDatabase('bolt://localhost:7687', 'neo4j', '123')
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
        dataset = read_dataset(dirpath)
        count = 1
        total_count = len(dataset)
        # Iterate all file records of the dataset.
        for filename, file in dataset:
            # Print the number of the currently processed file.
            print(f'Processing {count} out of {total_count} files...' )
            # Generate the terms from the text of each file.
            words = generate_words(file, extend_window, remove_stopwords, lemmatize, stem)
            # Create the graph of words in the database.
            value = create_graph_of_words(words, database, filename, window_size)
            if value is not None:
                skipped.append(value)
            # Update the progress counter.
            count = count + 1
            # Clear the screen to output the update the progress counter.
            clear_screen(current_system)
        # Count all skipped files and write their filenames in skipped.log
        skip_count = len(skipped)
        print(f'Created {total_count - skip_count}, skipped {skip_count} files.')
        print('Check skipped.log for info.')
        with open('skipped.log', 'w') as log:  
            for item in skipped:
                log.write(item + '\n')

    if initialize:
        # Run initialization functions.
        run_initial_algorithms(database)
        create_similarity_graph(database)
        create_clustering_tags(database)

    database.close()
    return

if __name__ == '__main__': 
    # If only one argument is specified,
    # Then it's the script name.
    # Print help for using the script and exit.
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    # Parse all arguments from terminal.
    args = parser.parse_args()

    # If create flag is set but no dirpath is specified, print error.
    if args.create and args.dirpath is None:
        parser.error('Please set the dirpath flag and specify a valid filepath!')
    # Else if create flag is specified along with a valid dirpath.
    elif args.create:
        print(args)
        # Run the graphofdocs function with create and initialize set to True.
        # The first argument (0th index) after the dirpath flag is the actual directory path. 
        graphofdocs(True, True, args.dirpath[0], args.window_size[0],
                   args.extend_window, args.insert_stopwords, args.lemmatize, args.stem)
    # Else if reinitialize flag is specified, unset the create flag.
    elif args.reinitialize:
        print(args)
        # Run the graphofdocs function with create set to False and initialize set to True.
        # We also set the directory path to None, since its not needed.
        graphofdocs(False, True, None, args.window_size[0],
                   args.extend_window, args.insert_stopwords, args.lemmatize, args.stem)
