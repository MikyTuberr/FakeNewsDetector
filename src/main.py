from DataParser import DataParser
from Serializer import Serializer
from NaiveBayesClassifier import NaiveBayesClassifier
from ErrorCalculator import ErrorCalculator
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import time

TRAIN_PATH = "../data/WELFake_Dataset.csv"
TEST_PATH = "../data/fake_or_real_news.csv"
TEXT_DATA_PATH = "../serialized/occur.pkl"
X_TRAIN_PATH = "../serialized/X_train.pkl"
X_TEST_PATH = "../serialized/X_test.pkl"
OWN_BAYES_PLOT_PATH = "../plots/own_bayes.png"
SKLEARN_PLOT_PATH = "../plots/sklearn.png"


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

    start_time = time.time()

    classifier.train(words_occur_fn, words_occur_rn, count_fn, count_rn, words_count_fn, words_count_rn)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"[Own Bayes] Training time: {training_time} seconds")

    test_data = DataParser.read_data(TEST_PATH)

    start_time = time.time()

    result = classifier.classify(DataParser.tokenize_data(test_data, "text", "title"))

    end_time = time.time()
    classifying_time = end_time - start_time
    print(f"[Own Bayes] Classifying time: {training_time} seconds")

    error_calculator = ErrorCalculator()
    accuracy = error_calculator.calc_accuracy(result, test_data["label"].values)
    print("[Own Bayes] Accuracy: " + str(accuracy))

    confusion_matrix = error_calculator.get_confusion_matrix(result, test_data["label"].values)
    print(confusion_matrix)
    error_calculator.show_confusion_matrix(confusion_matrix, OWN_BAYES_PLOT_PATH)


def sklearn_bayes_classifier():
    train_data = DataParser.read_data(TRAIN_PATH)
    X_train = [[]]
    if os.path.exists(X_TRAIN_PATH):
        X_train = Serializer.load_tokenized_X(X_TRAIN_PATH)
    else:
        X_train = DataParser.tokenize_data(train_data, "text", "title")
        Serializer.save_tokenized_X(X_TRAIN_PATH, X_train)

    test_data = DataParser.read_data(TEST_PATH)
    X_test = [[]]
    if os.path.exists(X_TEST_PATH):
        X_test = Serializer.load_tokenized_X(X_TEST_PATH)
    else:
        X_test = DataParser.tokenize_data(test_data, "text", "title")
        Serializer.save_tokenized_X(X_TEST_PATH, X_test)

    vectorizer = CountVectorizer(max_features=20, stop_words='english')
    X_train_counts = vectorizer.fit_transform([" ".join(text) for text in X_train])
    y_train = train_data["label"].values

    model = MultinomialNB()

    start_time = time.time()
    model.fit(X_train_counts, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"[Sklearn Bayes] Training time: {training_time} seconds")

    X_test_counts = vectorizer.transform([" ".join(text) for text in X_test])
    y_test = test_data["label"].values
    y_test = [1 if label == "REAL" else 0 for label in y_test]

    start_time = time.time()
    y_pred = model.predict(X_test_counts)
    end_time = time.time()
    classifying_time = end_time - start_time
    print(f"[Sklearn Bayes] Classifying time: {training_time} seconds")

    accuracy = accuracy_score(y_test, y_pred)
    print("[Sklearn Bayes] Accuracy: " + str(accuracy))

    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    error_calculator = ErrorCalculator()
    error_calculator.show_confusion_matrix(matrix, SKLEARN_PLOT_PATH)


def main() -> None:
    own_bayes_classifier()
    sklearn_bayes_classifier()


if __name__ == '__main__':
    main()
