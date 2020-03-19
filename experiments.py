import warnings
warnings.warn = lambda *args, **kwards: None # Supress warnings.

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from neo4j import ServiceUnavailable
from GraphOfDocs.neo4j_wrapper import Neo4jDatabase
from GraphOfDocs import utils
from GraphOfDocs import select
from prettytable import PrettyTable
from GraphOfDocs import config_experiments
from GraphOfDocs import evaluation
import timeit

results_table = PrettyTable(
    ['Method', 'Accuracy', 'Number of features',
    'Train size', 'Test size', 'Details'])
evaluation_results = []
feature_selection_evaluation_results = []

start_time = timeit.default_timer()
print('')
print('%'*100)
print('!START OF THE EXPERIMENT!')
#print('DATASET DIR PATH: %s' % config_experiments.DATASET_PATH)
print(f'DATASET DIR PATH: {config_experiments.DATASET_PATH}')
print('MIN NUMBER OF DOCUMENTS PER SELECTED COMMUNITY: '
      f'{config_experiments.MIN_NUMBER_OF_DOCUMENTS_PER_SELECTED_COMMUNITY}')
print('VARIANCE THRESHOLD: ' 
      f'{config_experiments.VARIANCE_THRESHOLD}')
print('SELECT KBEST K: '
      f'{config_experiments.SELECT_KBEST_K}')
print('TOP N SELECTED COMMUNITY TERMS: '
      f'{config_experiments.TOP_N_SELECTED_COMMUNITY_TERMS}')

# Connect to database.
try:
    database = Neo4jDatabase('bolt://localhost:7687', 'neo4j', '123')
    # Neo4j server is unavailable.
    # This client app cannot open a connection.
except ServiceUnavailable as error:
    print('\t* Neo4j database is unavailable.')
    print('\t* Please check the database connection before running this app.')
    input('\t* Press any key to exit the app...')
    sys.exit(1)
# Retrieve the communities of documents and their filenames.
doc_communities = select.get_communities_filenames(database)
# Keep only the communities with more than one documents.
filtered_doc_communities = \
    [doc_community for doc_community in doc_communities 
     if doc_community[2] >= 
     config_experiments.MIN_NUMBER_OF_DOCUMENTS_PER_SELECTED_COMMUNITY]
# Fetch the selected documents.
selected_docs = sum([docs for _, docs, _ in filtered_doc_communities], [])
# Map community id to documents.
doc_communities_dict = {community_id: docs 
                        for community_id, docs, number_of_docs 
                        in filtered_doc_communities}
# Map document to community id.
doc_to_community_dict = {doc: community_id 
                         for community_id, doc_community, _ 
                         in filtered_doc_communities for doc in doc_community}
print(f'Number of selected documents: {len(selected_docs)}')
# Read dataset, clean dataset and create a pandas dataframe of the dataset.
dataset = utils.read_dataset(config_experiments.DATASET_PATH)
# Create a label encoder (map classes to integer numbers).
le = LabelEncoder()
# The class of each document can be found by simply split (character '_') its filename. E.g. comp.sys.mac.hardware_51712.
le.fit([config_experiments.extract_file_class(file[0]) for file in dataset])
# Tuple: file identifier, file class, file class number, file text.
clean_dataset = [(file[0], 
                  config_experiments.extract_file_class(file[0]), 
                  le.transform([config_experiments.extract_file_class(file[0])])[0],
                  ' '.join(utils.generate_words(
                      file[1], 
                      extend_window = True, 
                      insert_stopwords = False, 
                      lemmatize = False, stem = False))) 
                  for file in dataset]
df = pd.DataFrame(clean_dataset, 
                  columns = ['identifier', 'class', 'class_number', 'text'])
df_all = df

# Keep only the selected documents (i.e. the document from the community with more than 1 documents).
df = df[df['identifier'].isin(selected_docs)]
df = shuffle(df, random_state = 42)
print('EXAMPLE OF THE PANDAS DATAFRAME')
print(df.head(2))

# Number of unique classes
print(f'Number of unique classes: {le.classes_.shape}')

X = df['text']
y = df['class_number']
positions = [i for i in range(len(X))]
positions_train, positions_test = train_test_split(
    positions, test_size  =0.33, random_state = 42)

res = evaluation.BOWEvaluator()\
    .evaluate(X, y, results_table = results_table,
             classifiers = config_experiments.classifiers)
evaluation_results.extend(res)

res = evaluation.MetaFeatureSelectionEvaluator()\
    .evaluate(X, y, results_table = results_table, 
              classifiers = config_experiments.classifiers)
evaluation_results.extend(res)

for variance_threshold in config_experiments.VARIANCE_THRESHOLD:
    res = evaluation\
        .LowVarianceFeatureSelectionEvaluator(
            variance_threshold=variance_threshold)\
                .evaluate(X, y, results_table = results_table,
                         classifiers = config_experiments.classifiers)
    evaluation_results.extend(res)
    feature_selection_evaluation_results.extend(res)

for kbest_k in config_experiments.SELECT_KBEST_K:
    res = evaluation\
        .SelectKBestFeatureSelectionEvaluator(
            kbest=kbest_k)\
                .evaluate(X, y, results_table = results_table,
                         classifiers=config_experiments.classifiers)
    evaluation_results.extend(res)
    feature_selection_evaluation_results.extend(res)

evaluation.GraphOfDocsClassifier(
    doc_to_community_dict, doc_communities_dict)\
        .calculate_accuracy(df['identifier'], results_table = results_table)

for top_n in config_experiments.TOP_N_SELECTED_COMMUNITY_TERMS:
    res = evaluation\
        .TopNOfEachCommunityEvaluator(
            top_n, doc_to_community_dict,
            doc_communities_dict)\
                .evaluate(
                    X, y, df = df, 
                    positions_train = positions_train, 
                    database = database, 
                    results_table = results_table, 
                    classifiers = config_experiments.classifiers)
    evaluation_results.extend(res)
    feature_selection_evaluation_results.extend(res)

df_evaluation_results = pd.DataFrame(evaluation_results)
df_feature_selection_evaluation_results = \
    pd.DataFrame(feature_selection_evaluation_results)
print('EXAMPLE OF THE EVALUATION RESULTS PANDAS DATAFRAME')
print(df_evaluation_results.head(2))

results_table.sortby = 'Accuracy'
results_table.reversesort = True
print(results_table)

output_dir = config_experiments.EXPERIMENTAL_RESULTS_OUΤPUT_DIR
plots_prefix = config_experiments.PLOTS_PREFIX
df_evaluation_results.to_csv(f'{output_dir}/{plots_prefix}_evaluation_results.csv')
evaluation.generate_plots(
    df_feature_selection_evaluation_results,
    output_dir = output_dir,
    plots_prefix = f'{plots_prefix}_feature_selection',
    show_only = False)

database.close()
stop_time = timeit.default_timer()
print(f'Execution time: {stop_time - start_time}')
print('!END OF THE EXPERIMENT!')
print('%'*100)
print('')
