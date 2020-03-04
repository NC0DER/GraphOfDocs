from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from collections import Counter
from GraphOfDocs import select
from config_experiments import extract_file_class

def benchmark_classifier(clf, X_train, y_train, X_test, y_test, round_digits=4):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, round_digits)
    return clf, accuracy

class GraphOfDocsClassifier:
    def __init__(self, doc_to_community_dict, doc_communities_dict, test_size=0.33, random_state=42):
        self.__test_size = test_size
        self.__random_state = random_state
        self.__doc_to_community_dict = doc_to_community_dict
        self.__doc_communities_dict = doc_communities_dict

    def calculate_accuracy(self, document_identifiers, results_table):
        _, test_docs = train_test_split(document_identifiers, test_size=self.__test_size, random_state=self.__random_state)
        test_docs = list(test_docs)
        class_true = []
        class_pred = []
        for test_doc in test_docs:
            community_id = self.__doc_to_community_dict[test_doc]
            community_docs = self.__doc_communities_dict[community_id]
            classes = [extract_file_class(doc) for doc in community_docs if doc != test_doc]

            correct_class = extract_file_class(test_doc)
            classified_class = Counter(classes).most_common(1)[0][0]
            class_true.append(correct_class)
            class_pred.append(classified_class)
        accuracy = round(accuracy_score(class_true, class_pred), 4)
        #print('Accuracy: %s' % (accuracy))
        results_table.add_row(['Graph-of-docs Classifier', accuracy, 'N/A', 'N/A', len(test_docs), ''])

class Evaluator:
    def __init__(self, test_size=0.33, random_state=42):
        self._test_size = test_size
        self._random_state = random_state
    def evaluate(self, x, y, **kwargs):
        raise NotImplemented('pure virtual')

    def _collect_evaluation_results(self, x_train_transformed, y_train, x_test_transformed, y_test, results_table, classifiers, method_prefix, extra_details=''):
        train_size = x_train_transformed.shape[0]
        test_size = x_test_transformed.shape[0]
        number_of_features = x_test_transformed.shape[1]
        for classifier in classifiers:
            _, accuracy = benchmark_classifier(classifier[1], x_train_transformed, y_train, x_test_transformed, y_test)
            # print('classifier:%s %s %s' % (classifier[0], accuracy, number_of_features))
            results_table.add_row([method_prefix + classifier[0], accuracy, number_of_features, train_size, test_size, extra_details])

class BOWEvaluator(Evaluator):
    def __init__(self, test_size=0.33, random_state=42):
        Evaluator.__init__(self, test_size, random_state)

    def evaluate(self, x, y, **kwargs):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size, random_state=self._random_state)
        cv = CountVectorizer()
        x_train_transformed = cv.fit_transform(x_train)
        x_test_transformed = cv.transform(x_test)

        results_table = kwargs['results_table']
        classifiers = kwargs['classifiers']
        self._collect_evaluation_results(x_train_transformed, y_train, x_test_transformed, y_test, results_table, classifiers, method_prefix='BOW + ')

class MetaFeatureSelectionEvaluator(Evaluator):
    def __init__(self, estimator_model=LinearSVC, test_size=0.33, random_state=42):
        Evaluator.__init__(self, test_size, random_state)
        self.__estimator_model = estimator_model

    def evaluate(self, x, y, **kwargs):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size, random_state=self._random_state)
        cv = CountVectorizer()
        x_train_transformed = cv.fit_transform(x_train)
        x_test_transformed = cv.transform(x_test)
        selector = SelectFromModel(estimator=self.__estimator_model())
        x_train_transformed = selector.fit_transform(x_train_transformed, y_train)
        x_test_transformed = selector.transform(x_test_transformed)

        results_table = kwargs['results_table']
        classifiers = kwargs['classifiers']
        self._collect_evaluation_results(x_train_transformed, y_train, x_test_transformed, y_test, results_table, classifiers, method_prefix='META + ')

class LowVarianceFeatureSelectionEvaluator(Evaluator):
    def __init__(self, variance_threshold, test_size=0.33, random_state=42):
        Evaluator.__init__(self, test_size, random_state)
        self.__variance_threshold = variance_threshold

    def evaluate(self, x, y, **kwargs):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size, random_state=self._random_state)
        cv = CountVectorizer()
        x_train_transformed = cv.fit_transform(x_train)
        x_test_transformed = cv.transform(x_test)
        selector = VarianceThreshold(threshold=self.__variance_threshold)
        x_train_transformed = selector.fit_transform(x_train_transformed, y_train)
        x_test_transformed = selector.transform(x_test_transformed)

        results_table = kwargs['results_table']
        classifiers = kwargs['classifiers']
        extra_details = 'variance thershold:' + str(self.__variance_threshold)
        self._collect_evaluation_results(x_train_transformed, y_train, x_test_transformed, y_test, results_table, classifiers, method_prefix='LOW VARIANCE + ', extra_details=extra_details)

class SelectKBestFeatureSelectionEvaluator(Evaluator):
    def __init__(self, kbest, test_size=0.33, random_state=42):
        Evaluator.__init__(self, test_size, random_state)
        self.__kbest = kbest

    def evaluate(self, x, y, **kwargs):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size, random_state=self._random_state)
        cv = CountVectorizer()
        x_train_transformed = cv.fit_transform(x_train)
        x_test_transformed = cv.transform(x_test)
        selector = SelectKBest(chi2, k=self.__kbest)
        x_train_transformed = selector.fit_transform(x_train_transformed, y_train)
        x_test_transformed = selector.transform(x_test_transformed)

        results_table = kwargs['results_table']
        classifiers = kwargs['classifiers']
        extra_details = 'kbest:' + str(self.__kbest)
        self._collect_evaluation_results(x_train_transformed, y_train, x_test_transformed, y_test, results_table, classifiers, method_prefix='KBEST + ', extra_details=extra_details)

class BigramsExtractionEvaluator(Evaluator):
    def __init__(self, test_size=0.33, random_state=42):
        Evaluator.__init__(self, test_size, random_state)

    def evaluate(self, x, y, **kwargs):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size, random_state=self._random_state)
        cv = CountVectorizer(ngram_range=(2,2))
        x_train_transformed = cv.fit_transform(x_train)
        x_test_transformed = cv.transform(x_test)

        results_table = kwargs['results_table']
        classifiers = kwargs['classifiers']
        self._collect_evaluation_results(x_train_transformed, y_train, x_test_transformed, y_test, results_table, classifiers, 'BIGRAMS + ')

class BigramsExtractionAndSelectKBestFeatureSelectionEvaluator(Evaluator):
    def __init__(self, kbest, test_size=0.33, random_state=42):
        Evaluator.__init__(self, test_size, random_state)
        self.__kbest = kbest

    def evaluate(self, x, y, **kwargs):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size, random_state=self._random_state)
        cv = CountVectorizer(ngram_range=(2,2))
        x_train_transformed = cv.fit_transform(x_train)
        x_test_transformed = cv.transform(x_test)
        selector = SelectKBest(chi2, k=self.__kbest)
        x_train_transformed = selector.fit_transform(x_train_transformed, y_train)
        x_test_transformed = selector.transform(x_test_transformed)

        results_table = kwargs['results_table']
        classifiers = kwargs['classifiers']
        extra_details = 'kbest:' + str(self.__kbest)
        self._collect_evaluation_results(x_train_transformed, y_train, x_test_transformed, y_test, results_table, classifiers, 'BIGRAMS + KBEST + ', extra_details=extra_details)

class TopNOfEachCommunityEvaluator(Evaluator):
    def __init__(self, top_n, doc_to_community_dict, doc_communities_dict, test_size=0.33, random_state=42):
        Evaluator.__init__(self, test_size, random_state)
        self.__top_n = top_n
        self.__doc_to_community_dict = doc_to_community_dict
        self.__doc_communities_dict = doc_communities_dict

    def evaluate(self, x, y, **kwargs):
        df = kwargs['df']
        positions_train = kwargs['positions_train']
        train_docs = list(df.iloc[positions_train]['identifier'])
        database = kwargs['database']
        vocabulary = []
        for doc in train_docs:
            for word in select.get_community_tags(database, self.__doc_to_community_dict[doc], top_terms=self.__top_n):
                vocabulary.append(word)
        vocabulary = list(set(vocabulary))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size, random_state=self._random_state)
        cv = CountVectorizer(vocabulary=vocabulary)
        x_train_transformed = cv.fit_transform(x_train, y_train)
        x_test_transformed = cv.transform(x_test)

        results_table = kwargs['results_table']
        classifiers = kwargs['classifiers']
        extra_details = 'top_n:' + str(self.__top_n)
        self._collect_evaluation_results(x_train_transformed, y_train, x_test_transformed, y_test, results_table, classifiers, 'TOP N WORD COMMUNITY + ', extra_details=extra_details)

# Ignore this class for the AIAI paper. Future work.
class Docs2ComEvaluator(Evaluator):
    def __init__(self, top_n, doc_to_community_dict, doc_communities_dict, label_encoder, test_size=0.33, random_state=42):
        Evaluator.__init__(self, test_size, random_state)
        self.__top_n = top_n
        self.__doc_to_community_dict = doc_to_community_dict
        self.__doc_communities_dict = doc_communities_dict
        self.__label_encoder = label_encoder

    # ### [tag1, tag2, ... tagN] -> class (Do this for each community of docs)
    # TODO: Clean up this method.
    def evaluate(self, x, y, **kwargs):
        df = kwargs['df']
        positions_train = kwargs['positions_train']
        positions_test = kwargs['positions_test']
        train_docs = list(df.iloc[positions_train]['identifier'])
        test_docs = list(df.iloc[positions_test]['identifier'])
        database = kwargs['database']
        unique_community_ids = list(set([self.__doc_to_community_dict[doc] for doc in train_docs]))

        communities_y = []
        communities_tags = []
        for community_id in unique_community_ids:
            # Find the most common community class
            community_docs = self.__doc_communities_dict[community_id]
            classes = [extract_file_class(doc) for doc in community_docs if doc not in test_docs]
            classified_class = Counter(classes).most_common(1)[0][0]
            communities_y.append(classified_class)
            # Get the most important tags of each community.
            communities_tags.append(' '.join(select.get_community_tags(database, community_id, top_terms=self.__top_n)))

        cv = CountVectorizer()
        x_transformed = cv.fit_transform(communities_tags)
        communities_y_encoded = self.__label_encoder.transform(communities_y)
        x_test_docs = []
        for doc in list(df[df['identifier'].isin(test_docs)]['text']):
            x_test_docs.append(' '.join(list(set(doc.split()))))
        x_test_docs_transformed = cv.transform(x_test_docs)
        y_test = list(df[df['identifier'].isin(test_docs)]['class_number'])

        results_table = kwargs['results_table']
        classifiers = kwargs['classifiers']
        extra_details = 'top_n:' + str(self.__top_n)
        self._collect_evaluation_results(x_transformed, communities_y_encoded, x_test_docs_transformed, y_test, results_table, classifiers, 'DOC2COM + ', extra_details=extra_details)

class GraphOfDocsBigramsExtractionEvaluator(Evaluator):
    def __init__(self, top_n=None, min_weight=None, test_size=0.33, random_state=42):
        Evaluator.__init__(self, test_size, random_state)
        self.__top_n = top_n
        self.__min_weight = min_weight

    def __generate_bigram_features(self, document_bigrams):
        if self.__top_n is not None:
            document_bigrams = document_bigrams[:self.__top_n]
        elif self.__min_weight is not None:
            document_bigrams = [bigram for bigram in document_bigrams if bigram[3] >= self.__min_weight]
        generated_bigrams = []
        for bigram in document_bigrams:
            generated_bigrams.append(bigram[0] + '-' + bigram[1])
        return generated_bigrams

    def __convert_documents_to_bigrams_dicts(self, database, document_ids):
        bigrams_dicts = []
        for document_id in document_ids:
            bigrams = select.get_word_digrams_by_filename(database, document_id)[0][0]
            bigrams = self.__generate_bigram_features(bigrams)
            bigrams_dicts.append({bigram: 1 for bigram in bigrams})
        return bigrams_dicts

    def evaluate(self, x, y, **kwargs):
        df = kwargs['df']
        database = kwargs['database']
        train_docs, test_docs, y_train, y_test = train_test_split(df['identifier'], y, test_size=self._test_size, random_state=self._random_state)
        train_docs = list(train_docs)
        test_docs = list(test_docs)

        train_documents = self.__convert_documents_to_bigrams_dicts(database, train_docs)
        test_documents = self.__convert_documents_to_bigrams_dicts(database, test_docs)

        dict_vectorizer = DictVectorizer()
        train_transformed = dict_vectorizer.fit_transform(train_documents)
        test_transformed = dict_vectorizer.transform(test_documents)

        results_table = kwargs['results_table']
        classifiers = kwargs['classifiers']
        extra_details = 'top_n:%s, min_weight:%s' %(self.__top_n, self.__min_weight)
        self._collect_evaluation_results(train_transformed, y_train, test_transformed, y_test, results_table, classifiers, 'Graph-of-docs BIGRAMS + ', extra_details=extra_details)