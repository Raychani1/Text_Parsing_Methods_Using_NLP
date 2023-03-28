import os
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

    @staticmethod
    def __draw_data_plot(
            data: pd.DataFrame,
            columns: List[str],
            title: str,
            path: str
    ) -> None:
        # TODO - Docstring

        # Create Plot
        plt.figure(figsize=(16, 9))

        # Draw Plot
        data[columns].plot()

        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Value')

        plt.draw()
        plt.draw()
        plt.savefig(path)

    def display_training_history(
        self, 
        history: pd.DataFrame, 
        index_col: str,
        path: str,
        timestamp: str
    ) -> None:
        """Displays Training History Plots.

        Args:
            history (pandas.DataFrame) : Training History Data.
            index_col (str) : Training History Data index column name.
            path (str): Training History plot output folder path.
            timestamp (str): Training History timestamp for output file name.
        """
        column_pairs = [
            ['train_loss', 'eval_loss'],
            ['train_precision', 'eval_precision'],
            ['train_recall', 'eval_recall'],
            ['train_accuracy', 'eval_accuracy'],
            ['train_f1', 'eval_f1']
        ]

        for column_pair in column_pairs:
            metric = column_pair[0].split('_')[-1]

            self.__draw_data_plot(
                data=history.set_index(index_col), 
                columns=column_pair,
                title=f'Training History - {metric.capitalize()}',
                path=os.path.join(
                    path, 
                    f'training_history_{metric}_{timestamp}.png'
                )
            )
