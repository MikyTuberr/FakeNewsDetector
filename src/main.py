from DataParser import DataParser
from Serializer import Serializer
from NaiveBayesClassifier import NaiveBayesClassifier
from ErrorCalculator import ErrorCalculator
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import numpy as np


TRAIN_PATH = "../data/WELFake_Dataset.csv"
TEST_PATH = "../data/fake_or_real_news.csv"
TEXT_DATA_PATH = "../data/occur.pkl"
X_TRAIN_PATH = "../data/X_train.pkl"
X_TEST_PATH = "../data/X_test.pkl"


def own_bayes_classifier():
    train_data = DataParser.read_data(TRAIN_PATH)

    if os.path.exists(TEXT_DATA_PATH):
        words_occur_fn, words_occur_rn, count_fn, count_rn, words_count_fn, words_count_rn \
            = Serializer.load_word_occurences(TEXT_DATA_PATH)
    else:
        words_occur_fn, words_occur_rn, count_fn, count_rn, words_count_fn, words_count_rn \
            = DataParser.count_occurrences(train_data, "text", "title", "label")
        Serializer.save_word_occurences(words_occur_fn, words_occur_rn, count_fn, count_rn,
                                        words_count_fn, words_count_rn, TEXT_DATA_PATH)

    classifier = NaiveBayesClassifier()
    classifier.train(words_occur_fn, words_occur_rn, count_fn, count_rn, words_count_fn, words_count_rn)

    test_data = DataParser.read_data(TEST_PATH)
    result = classifier.classify(DataParser.tokenize_data(test_data, "text", "title"))

    error_calculator = ErrorCalculator()
    accuracy = error_calculator.calc_accuracy(result, test_data["label"].values)
    print("Accuracy: " + str(accuracy))

    confusion_matrix = error_calculator.get_confusion_matrix(result, test_data["label"].values)
    print(confusion_matrix)
    error_calculator.show_confusion_matrix(confusion_matrix)


def sklearn_bayes_classifier():
    train_data = DataParser.read_data(TRAIN_PATH)
    X_train = [[]]
    if os.path.exists(X_TRAIN_PATH):
        X_train = Serializer.load_tokenized_X(X_TRAIN_PATH)
    else:
        X_train = DataParser.tokenize_data(train_data, "text", "title")
        Serializer.save_tokenized_X(X_TRAIN_PATH, X_train)

    vectorizer = CountVectorizer(max_features=30000)
    X_train_counts = vectorizer.fit_transform([" ".join(text) for text in X_train])
    y_train = train_data["label"].values

    model = MultinomialNB()
    model.fit(X_train_counts, y_train)

    test_data = DataParser.read_data(TEST_PATH)
    X_test = [[]]
    if os.path.exists(X_TEST_PATH):
        X_test = Serializer.load_tokenized_X(X_TEST_PATH)
    else:
        X_test = DataParser.tokenize_data(test_data, "text", "title")
        Serializer.save_tokenized_X(X_TEST_PATH, X_test)

    X_test_counts = vectorizer.fit_transform([" ".join(text) for text in X_test])
    y_test = test_data["label"].values
    y_test = [1 if label == "REAL" else 0 for label in y_test]
    y_pred = model.predict(X_test_counts)

    accuracy = accuracy_score(y_test, y_pred)
    print("Dokładność: {:.2f}".format(accuracy))

    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    error_calculator = ErrorCalculator()
    error_calculator.show_confusion_matrix(matrix)


def main() -> None:
    #own_bayes_classifier()
    sklearn_bayes_classifier()

if __name__ == '__main__':
    main()
