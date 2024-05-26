import pickle


class Serializer:
    def __init__(self):
        pass

    @staticmethod
    def save_word_occurences(words_occur_fn: dict[str, int], words_occur_rn: dict[str, int], count_fn: int,
                              count_rn: int, words_count_fn: int, words_count_rn: int, filename: str) -> None:
        try:
            with open(filename, 'wb') as file:
                pickle.dump(words_occur_fn, file)
                pickle.dump(words_occur_rn, file)
                pickle.dump(count_fn, file)
                pickle.dump(count_rn, file)
                pickle.dump(words_count_fn, file)
                pickle.dump(words_count_rn, file)
        except Exception as e:
            raise ("Error during saving word frequencies", e)

    @staticmethod
    def load_word_occurences(filename: str) -> tuple[
            dict[str, int], dict[str, int], int, int, int, int]:
        try:
            with open(filename, 'rb') as file:
                words_occur_fn = pickle.load(file)
                words_occur_rn = pickle.load(file)
                count_fn = pickle.load(file)
                count_rn = pickle.load(file)
                words_count_fn = pickle.load(file)
                words_count_rn = pickle.load(file)
            return words_occur_fn, words_occur_rn, count_fn, count_rn, words_count_fn, words_count_rn

        except FileNotFoundError as e:
            raise ("Error during loading word frequencies:", e)

    @staticmethod
    def save_tokenized_X(filename: str, x_data: list[list[str]]) -> None:
        try:
            with open(filename, 'wb') as file:
                pickle.dump(x_data, file)
        except Exception as e:
            raise ("Error during saving tokenized data", e)

    @staticmethod
    def load_tokenized_X(filename: str) -> list[list[str]]:
        try:
            with open(filename, 'rb') as file:
                x_data = pickle.load(file)
            return x_data

        except FileNotFoundError as e:
            raise ("Error during loading tokenized data:", e)