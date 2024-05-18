import matplotlib.pyplot as plt


class ErrorCalculator:
    def __init__(self):
        pass

    def calc_accuracy(self, result: list[str], label: list[str]) -> float:
        good_results = 0
        for i in range(len(label)):
            if result[i] == label[i]:
                good_results += 1
        return good_results / len(label)

    def get_confusion_matrix(self, result: list[str], label: list[str]) -> list[list[int]]:
        confusion_matrix = [[0, 0, ], [0, 0]]
        for i in range(len(label)):
            if label[i] == "REAL" and result[i] == "REAL":
                confusion_matrix[0][0] += 1
            elif label[i] == "REAL" and result[i] == "FAKE":
                confusion_matrix[0][1] += 1
            elif label[i] == "FAKE" and result[i] == "REAL":
                confusion_matrix[1][0] += 1
            elif label[i] == "FAKE" and result[i] == "FAKE":
                confusion_matrix[1][1] += 1
        return confusion_matrix

    def show_confusion_matrix(self, confusion_matrix: list[list[int]]):
        labels = ['True Positive', 'False Negative', 'False Positive', 'True Negative']
        values = [confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]]
        plt.bar(labels, values, color=['green', 'red', 'red', 'green'])
        plt.xlabel('Classes')
        plt.ylabel('Counts')
        plt.title('Confusion Matrix')
        plt.show()
