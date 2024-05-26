from collections import defaultdict

import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')


class DataParser:
    def __init__(self):
        pass

    @staticmethod
    def read_data(filename: str) -> pd.DataFrame:
        try:
            return pd.read_csv(filename)
        except pd.errors.EmptyDataError as e:
            raise ("Error during reading data", e)

    @staticmethod
    def _update_word_occurrences(word_dict: dict, tokens: list):
        try:
            for word in tokens:
                word_dict[word] += 1
        except RuntimeError as e:
            raise e

    @staticmethod
    def count_occurrences(data: pd.DataFrame, text_col: str, title_col: str, label_col: str) -> tuple[
        dict[str, int], dict[str, int], int, int, int, int]:
        try:
            words_occur_fn = defaultdict(int)
            words_occur_rn = defaultdict(int)
            count_fn = 0
            count_rn = 0
            words_count_fn = 0
            words_count_rn = 0

            for text, title, label in zip(data[text_col], data[title_col], data[label_col]):
                tokens = word_tokenize(text + " " + title)
                if label == 0:
                    count_fn += 1
                    words_count_fn += len(words_occur_fn)
                    DataParser._update_word_occurrences(words_occur_fn, tokens)
                elif label == 1:
                    count_rn += 1
                    words_count_rn += len(words_occur_rn)
                    DataParser._update_word_occurrences(words_occur_rn, tokens)

            return dict(words_occur_fn), dict(words_occur_rn), count_fn, count_rn, words_count_fn, words_count_rn

        except RuntimeError as e:
            raise ("Error during counting occurrences in data", e)

    @staticmethod
    def tokenize_data(data: pd.DataFrame, text_col: str, title_col: str) -> list[list[str]]:
        tokens = []
        for text, title in zip(data[text_col], data[title_col]):
            tokens.append(word_tokenize(str(text) + " " + str(title)))
        return tokens
    """
    @staticmethod
    def calculate_word_frequency(data: pd.DataFrame, col_name: str) -> list[dict]:
        try:
            word_freq = []
            occ_dict = defaultdict(int)
            tf_idf = []
            N = len(data)

            for text in data[col_name]:
                tokens = word_tokenize(text)
                total_words = len(tokens)
                bag_of_words = Counter(tokens)
                word_freq.append({word: count / total_words for word, count in bag_of_words.items()})
                for word in set(tokens):
                    occ_dict[word] += 1

            for word_freq_dict in word_freq:
                tf_idf_dict = {}
                for word, tf in word_freq_dict.items():
                    tf_idf_dict[word] = math.log(N / occ_dict[word]) * tf
                tf_idf.append(tf_idf_dict)

            return tf_idf

        except RuntimeError as e:
            print("Error during calc of words frequencies", e)
            return []
    """
