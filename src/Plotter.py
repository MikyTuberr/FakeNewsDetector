import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass

    @staticmethod
    def show_confusion_matrix(confusion_matrix: list[list[int]], file_path: str = None) -> None:
        labels = ['RN Positive', 'RN Negative', 'FN Positive', 'FN Negative']
        values = [confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][1], confusion_matrix[1][0]]
        plt.bar(labels, values, color=['green', 'red', 'green', 'red'])
        plt.xlabel('Classes')
        plt.ylabel('Counts')
        plt.title('Confusion Matrix')
        if file_path:
            plt.savefig(file_path)
        plt.show()