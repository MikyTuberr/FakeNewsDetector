import math

class NaiveBayesClassifier:
    def __init__(self):
        self._words_probs_fn = {}
        self._words_probs_rn = {}
        self._prob_fn = 0
        self._prob_rn = 0
        self._count_fn = 0
        self._count_rn = 0
        self._vocab_size = 0

    def train(self, words_occur_fn: dict[str, int], words_occur_rn: dict[str, int],
                            count_fn: int, count_rn: int, words_count_fn: int, words_count_rn: int):
        try:
            self._vocab_size = words_count_fn + words_count_rn
            self._count_fn = count_fn
            self._count_rn = count_rn
            for word in words_occur_fn.keys():
                self._words_probs_fn[word] = words_occur_fn[word] / words_count_fn
            for word in words_occur_rn.keys():
                self._words_probs_rn[word] = words_occur_rn[word] / words_count_rn

            all_count = count_fn + count_rn
            self._prob_fn = count_fn + 1 / all_count + 1
            self._prob_rn = count_rn + 1 / all_count + 1
        except Exception as e:
            raise ("Error in NaiveBayesClassifier._calc_probabilities()", e)

    def classify(self, texts_tokens: list[list[str]]) -> list[str]:
        try:
            result = []
            for text in texts_tokens:
                fn_prob = math.log(self._prob_fn)
                rn_prob = math.log(self._prob_rn)
                for word in text:
                    if word in self._words_probs_fn:
                        fn_prob += math.log(self._words_probs_fn[word])
                    else:
                        fn_prob += math.log(1.0 / self._vocab_size + self._count_fn)
                    if word in self._words_probs_rn:
                        rn_prob += math.log(self._words_probs_rn[word])
                    else:
                        rn_prob *= math.log(1.0 / self._vocab_size + self._count_rn)
                if fn_prob > rn_prob:
                    result.append("FAKE")
                else:
                    result.append("REAL")
            return result
        except Exception as e:
            raise ("Error in NaiveBayesClassifier.classify", e)
