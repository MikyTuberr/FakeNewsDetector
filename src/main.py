from src.DataParser import DataParser
from Serializer import Serializer

DATA_PATH = "../data/fake_or_real_news.csv"
WORD_FREQ_PATH = "../data/word_frequencies.pkl"


def main() -> None:
    word_frequencies = Serializer.load_word_frequencies(WORD_FREQ_PATH)
    if len(word_frequencies) == 0:
        data = DataParser.read_data(DATA_PATH)
        word_frequencies = DataParser.calculate_word_frequency(data, "text")
        print(word_frequencies)
        Serializer.save_word_frequencies(word_frequencies, WORD_FREQ_PATH)
    print(word_frequencies)


if __name__ == '__main__':
    main()
