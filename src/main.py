from src.OwnNaiveBayesClassifier import OwnNaiveBayesClassifier
from Plotter import Plotter
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from BERTClassifier import BERTClassifier

TEST_DATA_PATH = "../data/fake_or_real_news.csv"
SERIALIZED_TEST_DATA_PATH = "../serialized/test_own_bayes.pkl"

TRAIN_DATA_PATH = "../data/WELFake_Dataset.csv"
SERIALIZED_TRAIN_DATA_PATH = "../serialized/train_own_bayes.pkl"

OWN_BAYES_PLOT_PATH = "../plots/own_bayes.png"
SKLEARN_PLOT_PATH = "../plots/sklearn.png"

# Load test data
#x_test, y_test = DataManager.load_or_preprocess_data(SERIALIZED_TEST_DATA_PATH, TEST_DATA_PATH, "text",
# "title", "label")
#y_test = [1 if label == 'REAL' else 0 for label in y_test]

# Load train data
#x, y = DataManager.load_or_preprocess_data(SERIALIZED_TRAIN_DATA_PATH, TRAIN_DATA_PATH, "text",
                                          # "title", "label")

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def bert_classifier() -> None:
    classifier = BERTClassifier("../data/WELFake_Dataset.csv",
                                "../serialized/bert_data.pkl", "../plots/bert.png",
                                256, 256)
    classifier.train(num_epochs=3, batch_size=8)
    classifier.evaluate(batch_size=8)


def own_bayes_classifier() -> None:
    # Initialize classifier
    classifier = OwnNaiveBayesClassifier(x_train, y_train, x_test, y_test)

    # Begin training
    start_time = time.time()

    classifier.train()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"[Own Bayes] Training time: {training_time} seconds")

    # Begin classification
    start_time = time.time()

    y_pred = classifier.classify(max_features=50000)

    end_time = time.time()
    classifying_time = end_time - start_time
    print(f"[Own Bayes] Classifying time: {classifying_time} seconds")

    # Get accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("[Own Bayes] Accuracy: " + str(accuracy))

    # Get confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    Plotter.show_confusion_matrix(matrix, OWN_BAYES_PLOT_PATH)


def sklearn_bayes_classifier():
    model = MultinomialNB()

    start_time = time.time()
    vectorizer = CountVectorizer(max_features=50000, stop_words='english', max_df=0.009, binary=True)
    X_train_counts = vectorizer.fit_transform([" ".join(text) for text in x_train])
    #print(vectorizer.get_feature_names_out())
    model.fit(X_train_counts, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"[Sklearn Bayes] Training time: {training_time} seconds")

    X_test_counts = vectorizer.transform([" ".join(text) for text in x_test])

    start_time = time.time()
    y_pred = model.predict(X_test_counts)
    end_time = time.time()
    classifying_time = end_time - start_time
    print(f"[Sklearn Bayes] Classifying time: {training_time} seconds")

    accuracy = accuracy_score(y_test, y_pred)
    print("[Sklearn Bayes] Accuracy: " + str(accuracy))

    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    Plotter.show_confusion_matrix(matrix, SKLEARN_PLOT_PATH)


def main() -> None:
    #own_bayes_classifier()
    #sklearn_bayes_classifier()
    bert_classifier()


if __name__ == '__main__':
    main()
