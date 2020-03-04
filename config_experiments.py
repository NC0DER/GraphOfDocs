MIN_NUMBER_OF_DOCUMENTS_PER_SELECTED_COMMUNITY = 2
DATASET_PATH = '/home/nkanak/Desktop/phd/experiments/GraphOfDocs/GraphOfDocs/data/20news-18828-all/'

PLOTS_PREFIX = '20NEWSGROUPS'
EXPERIMENTAL_RESULTS_OUPUT_DIR = 'experimental_results/20newsgroups'

# Feature selection
VARIANCE_THRESHOLD = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.01]
SELECT_KBEST_K = [1000, 2000, 3000, 5000, 10000, 15000, 20000, 25000, 30000]

# Graph of docs feature selection.
# Create a vocabulary with the TOP N words of each community of docs
TOP_N_SELECTED_COMMUNITY_TERMS = [25, 50, 100, 250, 500, 1000]
TOP_N_GRAPH_OF_DOCS_BIGRAMS = [5, 10, 15, 20, 30, 50, 70, 80, 100, 150, 200, 250, 300]
MIN_WEIGHT_GRAPH_OF_DOCS_BIGRAMS = [10, 20, 30, 50, 70, 80, 100, 150, 200, 250, 300]

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

classifiers = [
    ('NB', MultinomialNB()),
    ('LR', LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')),
    ('5NN', KNeighborsClassifier(n_neighbors=5, weights='distance')),
    ('2NN', KNeighborsClassifier(n_neighbors=2, weights='distance')),
    ('1KNN', KNeighborsClassifier(n_neighbors=1, weights='distance')),
    ('LSVM', LinearSVC()),
    ('NN100x50', MLPClassifier(solver='adam', hidden_layer_sizes=(100, 50), random_state=42)),
    ('NN500x250', MLPClassifier(solver='adam', hidden_layer_sizes=(500, 250), random_state=42)),
]

def extract_file_class(filename):
    return filename.split('_')[0]