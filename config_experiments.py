from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

MIN_NUMBER_OF_DOCUMENTS_PER_SELECTED_COMMUNITY = 2
DATASET_PATH = \
    r'C:\Users\USER\source\repos\GraphOfDocs\GraphOfDocs\datasets\amazon'

PLOTS_PREFIX = 'AMAZON'
EXPERIMENTAL_RESULTS_OUΤPUT_DIR = \
    r'C:\Users\USER\source\repos\GraphOfDocs\GraphOfDocs\experimental_results\amazon'

# Feature selection
VARIANCE_THRESHOLD = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.01]
SELECT_KBEST_K = [350, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]

# Graph of docs feature selection.
# Create a vocabulary with the TOP N words of each community of docs
TOP_N_SELECTED_COMMUNITY_TERMS = [5, 10, 15, 20, 25, 50, 100, 250, 500]

#VARIANCE_THRESHOLD = [0.0005]
#SELECT_KBEST_K = [1000]
#TOP_N_SELECTED_COMMUNITY_TERMS = [5]

classifiers = [
    ('NB', MultinomialNB()),
    ('LR', LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')),
    ('5NN', KNeighborsClassifier(n_neighbors=5, weights='distance')),
    ('2NN', KNeighborsClassifier(n_neighbors=2, weights='distance')),
    ('1KNN', KNeighborsClassifier(n_neighbors=1, weights='distance')),
    ('LSVM', LinearSVC()),
    ('NN100x50', MLPClassifier(solver='adam', hidden_layer_sizes=(100, 50), random_state=42)),
    #('NN500x250', MLPClassifier(solver='adam', hidden_layer_sizes=(500, 250), random_state=42)),
]

def extract_file_class(filename):
    return filename.split('_')[0].split('.')[1]
