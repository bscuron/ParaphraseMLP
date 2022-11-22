import pandas as pd
import numpy as np
import ssl
from nltk import download, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.translate import nist_score, bleu_score
import string
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
# from imblearn.over_sampling import SMOTE, ADASYN # TODO: fix imbalanced data
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from thefuzz.fuzz import ratio, partial_ratio, token_sort_ratio, token_set_ratio
from difflib import SequenceMatcher

DATA_TRAIN_PATH='../data/train_with_label.txt'                                                                                             # training set
DATA_DEV_PATH='../data/dev_with_label.txt'                                                                                                 # dev set
DATA_TEST_PATH='../data/test_without_label.txt'                                                                                            # test set
TMP_DIR = '../tmp'                                                                                                                         # temporary directory to store processed dataframes
TEST_PRED_FILE = '../BenjaminScuron_test_result.txt'                                                                                       # text file that stores the predicted values of the test set
FEATURE_COLUMNS = ['LEVENSHTEIN_DIST', 'COSINE_SIMILARITY', 'LENGTH_DIFFERENCE', 'SHARED_WORDS', 'SHARED_POS', 'NIST_SCORE'] # features used in the MLP model

def main():
    print('Reading, cleaning, extracting features...')
    data_train, data_dev, data_test = get_data()

    # dump dataframes
    print(f'Dumping dataframes to `{TMP_DIR}`...')
    data_train.to_csv(f'{TMP_DIR}/data_train_processed.csv')
    data_dev.to_csv(f'{TMP_DIR}/data_dev_processed.csv')
    data_test.to_csv(f'{TMP_DIR}/data_test_processed.csv')

    print('Printing extracted features...')
    print(data_train[FEATURE_COLUMNS + ['GROUND_TRUTH']])
    print(data_dev[FEATURE_COLUMNS + ['GROUND_TRUTH']])
    print(data_test[FEATURE_COLUMNS])

    # Create model and fit to training data
    print('Creating MLP model...')
    # clf = make_pipeline(StandardScaler(), MLPClassifier(random_state=1, max_iter=300)) # TODO: scale
    clf = MLPClassifier(random_state=1, max_iter=300)
    print('Fitting model to training data...')
    clf.fit(data_train[FEATURE_COLUMNS], data_train['GROUND_TRUTH'])

    # Compute accuracy on dev
    print('Making predictions on DEV set...')
    y_dev_pred = clf.predict(data_dev[FEATURE_COLUMNS])
    print('Computing DEV accuracy and f1 score...')
    print('DEV ACCURACY:', accuracy_score(data_dev['GROUND_TRUTH'], y_dev_pred))
    print('DEV F1 SCORE:', f1_score(data_dev['GROUND_TRUTH'], y_dev_pred))

    # Make predictions on test data and output the test data
    print('Making predictions on TEST set...')
    y_test_labels = data_test['ID']
    y_test_pred = clf.predict(data_test[FEATURE_COLUMNS])
    print(f'Writing predictions to `{TEST_PRED_FILE}`')
    with open(f'{TEST_PRED_FILE}', 'w') as f:
        for id, label in zip(y_test_labels, y_test_pred):
            print(f'{id}\t{label}', file=f)


# Read and clean the train set, dev set, and test set. Return each in a tuple in the order (train, dev, test)
def get_data():
    delimiter, column_names = '\t+', ['ID', 'SENTENCE_1', 'SENTENCE_2', 'GROUND_TRUTH']
    data_train = extract_features(clean(pd.read_csv(DATA_TRAIN_PATH, sep=delimiter, engine='python', names=column_names)))
    data_dev = extract_features(clean(pd.read_csv(DATA_DEV_PATH, sep=delimiter, engine='python', names=column_names)))
    data_test = extract_features(clean(pd.read_csv(DATA_TEST_PATH, sep=delimiter, engine='python', names=column_names[:-1])))
    return data_train, data_dev, data_test

# Cleans the text data in columns 'SENTENCE_1' and 'SENTENCE_2'
def clean(df):
    lemmatizer = WordNetLemmatizer()
    # ILLEGAL = set(stopwords.words('english') + list(string.punctuation))
    ILLEGAL = set(list(string.punctuation))
    df['SENTENCE_1'] = df['SENTENCE_1'].apply(lambda x: ' '.join([lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in word_tokenize(x) if token not in ILLEGAL and len(token.strip(string.punctuation)) > 1]))
    df['SENTENCE_2'] = df['SENTENCE_2'].apply(lambda x: ' '.join([lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in word_tokenize(x) if token not in ILLEGAL and len(token.strip(string.punctuation)) > 1]))
    return df

# TODO
def extract_features(df):
    df['LEVENSHTEIN_DIST'] = get_levenshtein_distance(df)
    df['LENGTH_DIFFERENCE'] = get_length_difference(df)
    df['COSINE_SIMILARITY'] = get_cosine_similarity(df)
    df['SHARED_WORDS'] = get_shared_words(df)
    # df['RATIO'] = get_ratios(df)
    # df['PARTIAL_RATIO'] = get_partial_ratios(df)
    # df['TOKEN_SORT_RATIO'] = get_token_sort_ratios(df)
    # df['TOKEN_SET_RATIO'] = get_token_set_ratios(df)
    # df['RATCLIFF_OBERSHELP'] = get_ratcliff_obershelp(df)
    # df['JACCARD_SIMILARITY'] = get_jaccard_similarity(df)
    df['SHARED_POS'] = get_shared_pos(df)
    df['NIST_SCORE'] = get_nist_score(df)
    # df['BLEU_SCORE'] = get_bleu_score(df)
    return df

# Calculates the nist score for each sentence. The average of the two scores are taken (s1 as reference, s2 as hypothesis & s2 as reference, s1 as hypothesis)
def get_nist_score(df):
    scores = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        scores.append((nist_score.sentence_nist([s1], s2) + nist_score.sentence_nist([s2], s1)) / 2)
    return scores

# Calculates the bleu score for each sentence. The average of the two scores are taken (s1 as reference, s2 as hypothesis & s2 as reference, s1 as hypothesis)
def get_bleu_score(df):
    scores = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        scores.append((bleu_score.sentence_bleu([s1], s2) + bleu_score.sentence_bleu([s2], s1)) / 2)
    return scores

# Returns the number of shared pos tags between 'SENTENCE_1' and 'SENTENCE_2'
def get_shared_pos(df):
    shared = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        l1 = [s[1] for s in pos_tag(word_tokenize(s1))]
        l2 = [s[1] for s in pos_tag(word_tokenize(s2))]

        d1 = {}
        for pos in l1:
            d1[pos] = 1 if pos not in d1 else d1[pos] + 1

        d2 = {}
        for pos in l2:
            d2[pos] = 1 if pos not in d2 else d2[pos] + 1

        s = []
        common_keys = set(d1.keys()).intersection(d2.keys())
        for key in common_keys:
            s.append(min(d1[key], d2[key]))
        shared.append(sum(s))

    return shared

# Calculates the jaccard similarities of columns 'SENTENCE_1' and 'SENTENCE_2'
def get_jaccard_similarity(df):
    similarities = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        w1l, w2l = s1.split(), s2.split()
        intersection = set(w1l).intersection(set(w2l))
        union = set(w1l).union(set(w2l))
        similarities.append(float(len(intersection)) / len(union))
    return similarities

# Calculates the ratcliff obershelp ratios of columns 'SENTENCE_1' and 'SENTENCE_2'
def get_ratcliff_obershelp(df):
    ratios = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        ratios.append(SequenceMatcher(None, s1, s2).ratio())
    return ratios

# Calculates the similarity ratios of columns 'SENTENCE_1' and 'SENTENCE_2'
def get_ratios(df):
    ratios = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        ratios.append(ratio(s1, s2))
    return ratios

# Calculates the similarity partial ratios of columns 'SENTENCE_1' and 'SENTENCE_2'
def get_partial_ratios(df):
    ratios = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        ratios.append(partial_ratio(s1, s2))
    return ratios

# Calculates the token sort similarity ratios of columns 'SENTENCE_1' and 'SENTENCE_2'
def get_token_sort_ratios(df):
    ratios = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        ratios.append(token_sort_ratio(s1, s2))
    return ratios

# Calculates the token set similarity ratios of columns 'SENTENCE_1' and 'SENTENCE_2'
def get_token_set_ratios(df):
    ratios = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        ratios.append(token_set_ratio(s1, s2))
    return ratios

# Calculates the number of shared words of columns 'SENTENCE_1' and 'SENTENCE_2'
def get_shared_words(df):
    shared_words_list = []
    for s1, s2, in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        d1 = {}
        for w in s1.split():
            d1[w] = 1 if w not in d1 else d1[w] + 1
        d2 = {}
        for w in s2.split():
            d2[w] = 1 if w not in d2 else d2[w] + 1
        shared_words = set(d1) & set(d2)
        d = {}
        for w in shared_words:
            d[w] = min(d1[w], d2[w])
        shared_words_list.append(sum(d.values()))
    return shared_words_list

# Calculates the cosine similarity of columns 'SENTENCE_1' and 'SENTENCE_2'
def get_cosine_similarity(df):
    vectorizer = TfidfVectorizer()
    corpus = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        corpus.append(s1)
        corpus.append(s2)
    vectorizer.fit(corpus)
    sentence_1_vectors = df['SENTENCE_1'].apply(lambda x: np.array([value for value in vectorizer.transform([x]).A]).flatten())
    sentence_2_vectors = df['SENTENCE_2'].apply(lambda x: np.array([value for value in vectorizer.transform([x]).A]).flatten())
    cosine_simililarity_vector = []
    for vec1, vec2 in zip(sentence_1_vectors, sentence_2_vectors):
        cosine_simililarity_vector.append(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0])
    return cosine_simililarity_vector

# Calculates the length difference of columns 'SENTENCE_1' and 'SENTENCE_2'
def get_length_difference(df):
    differences = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        differences.append(abs(len(s1) - len(s2)))
    return differences


# Calculates the levenshtein distance of columns 'SENTENCE_1' and 'SENTENCE_2'
def get_levenshtein_distance(df):
    distances = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        distances.append(levenshtein_distance(s1, s2))
    return distances

if __name__ == '__main__':
    # Disable SSL check
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Downlad stop words
    download('stopwords')

    # Download the tagger
    download('averaged_perceptron_tagger')

    download('punkt')

    # Run main
    main()
