from typing import Tuple, List, Dict, Any

import pandas as pd
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
import nltk
import math

nltk.download('punkt')


class DataParser:
    def __init__(self):
        pass

    @staticmethod
    def read_data(filename: str) -> pd.DataFrame:
        try:
            return pd.read_csv(filename)
        except pd.errors.EmptyDataError as e:
            print("Error during reading data", e)
            return pd.DataFrame()

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
