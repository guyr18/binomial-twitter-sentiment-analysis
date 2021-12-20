from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import re
import pandas as pd
from sklearn import metrics
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer


porter = PorterStemmer()
tok = WordPunctTokenizer()
# Regex to help with data cleaning
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))


# Beautiful Soup is a Python library for pulling data out of HTML and XML files.
# It works with your favorite parser to provide idiomatic ways of navigating, searching, and modifying the parse tree.
# Tweet_cleaner cleans the tweets using modern text preprocessing libraries.
# Tweet_cleaner returns the cleaned text as a list of tokens.
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()

    # During the letters_only process two lines above, it has created unnecessary white spaces,
    # I will tokenize and join together to remove unnecessary white spaces
    words = tok.tokenize(lower_case)

    #Stemming
    stem_sentence = []
    for word in words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")

    words = "".join(stem_sentence).strip()

    return words


# Uses tweet_cleaner to clean the tweets as data_setup iterates through the csv file.
# This returns the cleaned test and train datasets
def data_setup(train, test):
    nums = [0, len(train)]
    clean_tweet_texts = []

    for i in range(nums[0], nums[1]):
        clean_tweet_texts.append(tweet_cleaner(train['tweet'][i]))

    nums = [0, len(test)]
    test_tweet_texts = []

    for i in range(nums[0], nums[1]):
        test_tweet_texts.append(tweet_cleaner(test['tweet'][i]))

    train_clean = pd.DataFrame(clean_tweet_texts, columns=['tweet'])
    train_clean['label'] = train.label
    train_clean['id'] = train.id
    test_clean = pd.DataFrame(test_tweet_texts, columns=['tweet'])
    test_clean['id'] = test.id

    return train_clean, test_clean


# The scores given are f1 scores
# f1 score is explained as 2*(Recall * Precision) / (Recall + Precision)
# recall is the ratio of correctly predicted positive observations to all observations in the class
# precision is the ratio of correctly predicted positive observations to the total predicted positive observations
# Train_model calculates and returns the f1 score of the dataset with the given classifier.
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    return metrics.f1_score(valid_y, predictions)


# Train_model_test does the same thing as train_model, except it doesn't have a list of correct labels to compare
# its predictions to.
def train_model_test(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return predictions


# data_extraction_train computes and returns parameters needed for resampling methods.
# It returns x_tfidf, y, xvalid_tfidf, valid_y.
# Each returned parameters is a list of the values in each column of the csv that the data is extracted from.
def data_extraction_train(data):
    # split the dataset into training and validation datasets
    x, valid_x, y, valid_y = model_selection.train_test_split(data['tweet'], data['label'])

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(y)
    valid_y = encoder.fit_transform(valid_y)

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100000)
    tfidf_vect.fit(data['tweet'])
    x_tfidf = tfidf_vect.transform(x)
    xvalid_tfidf = tfidf_vect.transform(valid_x)

    return x_tfidf, y, xvalid_tfidf, valid_y


# data_extraction_test computes and returns parameters needed for resampling methods.
# It returns x_tfidf, y, xvalid_tfidf.
# Each returned parameters is a list of the values in each column of the csv that the data is extracted from.
# The difference between data_extraction_train and *test is that test does not return a list of correct labels.
def data_extraction_test(train_clean, test_clean):
    train_x = train_clean['tweet']
    valid_x = test_clean['tweet']
    train_y = train_clean['label']

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)

    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english', max_features=100000)
    tfidf_vect.fit(train_clean['tweet'])
    tfidf_vect.fit(test_clean['tweet'])
    xtrain_tfidf = tfidf_vect.transform(train_x)
    xvalid_tfidf = tfidf_vect.transform(valid_x)

    return xtrain_tfidf, train_y, xvalid_tfidf, valid_x
