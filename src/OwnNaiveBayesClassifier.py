from collections import defaultdict
import nltk
import math
import random

class OwnNaiveBayesClassifier:
    def __init__(self, x_train: list[list[str]], y_train: list[int], x_test: list[list[str]], y_test: list[int]):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self._words_probs_fn = defaultdict(float)
        self._words_probs_rn = defaultdict(float)
        self._prob_fn = 0.0
        self._prob_rn = 0.0
        self._count_fn = 0
        self._count_rn = 0
        self._vocab_size = len(set(word for text in self.x_train for word in text))

        self._words_occur_fn = defaultdict(int)
        self._words_occur_rn = defaultdict(int)
        self._words_count_fn = 0
        self._words_count_rn = 0

    def _update_word_occurrences(self, word_dict: defaultdict, tokens: list):
        for word in tokens:
            word_dict[word] += 1

    def _count_occurrences(self) -> None:
        for text, label in zip(self.x_train, self.y_train):
            if label == 0:
                self._count_fn += 1
                self._words_count_fn += len(text)
                self._update_word_occurrences(self._words_occur_fn, text)
            elif label == 1:
                self._count_rn += 1
                self._words_count_rn += len(text)
                self._update_word_occurrences(self._words_occur_rn, text)

    def train(self):
        try:
            self._count_occurrences()
            unique_count_rn = len(set(self._words_occur_rn.keys()))
            unique_count_fn = len(set(self._words_occur_fn.keys()))

            for text in self.x_train:
                for word in text:
                    self._words_probs_fn[word] = ((self._words_occur_fn[word] + 1) /
                                                  (self._words_count_fn + unique_count_fn))

                    self._words_probs_rn[word] = ((self._words_occur_rn[word] + 1) /
                                                  (self._words_count_rn + unique_count_rn))

            all_count = self._count_fn + self._count_rn
            self._prob_fn = self._count_fn / all_count
            self._prob_rn = self._count_rn / all_count
        except Exception as e:
            raise RuntimeError("Error in NaiveBayesClassifier.train()", e)

    def classify(self, max_features: int) -> list[int]:
        try:
            result = []
            unique_count_rn = len(set(self._words_occur_rn.keys()))
            unique_count_fn = len(set(self._words_occur_fn.keys()))

            for text in self.x_test:
                fn_prob = math.log(self._prob_fn)
                rn_prob = math.log(self._prob_rn)

                # Sort words by their probabilities and select the top max_features
                sorted_words_fn = sorted(text, key=lambda word: self._words_probs_fn.get(word, 1 / (self._words_count_fn + unique_count_fn)), reverse=True)
                sorted_words_rn = sorted(text, key=lambda word: self._words_probs_rn.get(word, 1 / (self._words_count_rn + unique_count_rn)), reverse=True)
                words = sorted_words_fn[:max_features] + sorted_words_rn[:max_features]

                for word in words:
                    if word in self._words_probs_fn:
                        fn_prob += math.log(self._words_probs_fn[word])
                    else:
                        fn_prob += math.log(1 / (self._words_count_fn + unique_count_fn))
                    if word in self._words_probs_rn:
                        rn_prob += math.log(self._words_probs_rn[word])
                    else:
                        rn_prob += math.log(1 / (self._words_count_rn + unique_count_rn))

                if fn_prob > rn_prob:
                    result.append(0)
                else:
                    result.append(1)
            return result
        except Exception as e:
            raise RuntimeError("Error in NaiveBayesClassifier.classify()", e)

