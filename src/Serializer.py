import pickle


class Serializer:
    def __init__(self):
        pass

    @staticmethod
    def save_word_frequencies(word_frequencies: list, filename: str) -> None:
        try:
            with open(filename, 'wb') as file:
                pickle.dump(word_frequencies, file)
        except Exception as e:
            print("Error during saving word frequencies", e)

    @staticmethod
    def load_word_frequencies(filename: str) -> list:
        try:
            word_frequencies = []
            with open(filename, 'rb') as file:
                word_frequencies = pickle.load(file)
            return word_frequencies
        except Exception as e:
            print("Error during loading word frequencies:", e)
            return []

