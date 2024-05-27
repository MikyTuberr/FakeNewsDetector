import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
import pickle

nltk.download('punkt')
nltk.download('stopwords')


class DataManager:
    def __init__(self):
        pass

    @staticmethod
    def _load_data_form_csv(path: str):
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError as e:
            print("Error during loading data from file " + path, e)
            return pd.DataFrame()

    @staticmethod
    def _tokenize_data(data: pd.DataFrame, text_col: str, title_col: str, label_col: str) -> tuple[
        list[list[str]], list[int]]:
        stop_words = set(stopwords.words('english'))
        X = []
        Y = []

        for text, title, label in zip(data[text_col], data[title_col], data[label_col]):
            tokens = []

            if isinstance(text, str):
                text_tokens = word_tokenize(text)
                text_tokens = [word.lower() for word in text_tokens if
                               word.isalnum() and word.lower() not in stop_words]
                tokens.extend(text_tokens)

            if isinstance(title, str):
                title_tokens = word_tokenize(title)
                title_tokens = [word.lower() for word in title_tokens if
                                word.isalnum() and word.lower() not in stop_words]
                tokens.extend(title_tokens)

            if tokens:
                X.append(tokens)
                Y.append(label)

        return X, Y

    @staticmethod
    def _serialize_to_pickle(path: str, X: list[list[str]], Y: list[int]) -> None:
        try:
            with open(path, 'wb') as file:
                pickle.dump(X, file)
                pickle.dump(Y, file)
        except Exception as e:
            raise ("Error during serializing to " + path, e)

    @staticmethod
    def load_or_preprocess_data(serialized_path: str, unserialized_path: str, text_col: str, title_col: str, label_col: str):
        try:
            if os.path.exists(serialized_path):
                print("Loading " + serialized_path)
                with open(serialized_path, 'rb') as f:
                    x = pickle.load(f)
                    y = pickle.load(f)
                return x, y
            else:
                print("Starting to load data: " + unserialized_path)
                data = DataManager._load_data_form_csv(unserialized_path)
                print("Data loaded from csv: " + unserialized_path)
                print("Starting to tokenize: " + unserialized_path)
                X, Y = DataManager._tokenize_data(data, text_col, title_col, label_col)
                print("Data tokenized: " + unserialized_path)
                print("Starting to serialize data: " + unserialized_path)
                DataManager._serialize_to_pickle(serialized_path, X, Y)
                print("Data serialized: " + serialized_path)
                return X, Y
        except Exception as e:
            print("Error during processing data", e)
            return [], []
