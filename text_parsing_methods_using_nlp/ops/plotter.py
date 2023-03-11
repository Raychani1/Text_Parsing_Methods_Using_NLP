from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class Plotter:

    @staticmethod
    def display_confusion_matrix(
        conf_matrix: np.ndarray,
        title: str,
        labels: List[Any],
        path: str
    ) -> None:
        """Displays the passed Confusion Matrix.

        Args:
            confusion_matrix (numpy.ndarray): Confusion Matrix to display.
            title (str): Plot title.
            labels (List[Any]): Labels for Columns and Indexes.
            path (str): Output save path.
        """
        # SOURCE:
        # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix

        # Convert Confusion Matrix to DataFrame
        conf_matrix = pd.DataFrame(
            data=conf_matrix,
            index=labels,
            columns=labels
        )

        # Create new Figure
        figure = plt.figure(figsize=(16, 9))

        # Add simple axes to plot on
        ax = figure.add_subplot(1, 1, 1)

        # Plot the Confusion Matrix
        sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')

        plt.draw()
        plt.savefig(path)
        plt.show(block=False)