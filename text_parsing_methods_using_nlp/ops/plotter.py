import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import seaborn as sns
from natsort import natsorted
from pandas.api.types import CategoricalDtype

from text_parsing_methods_using_nlp.config.config import (
    EVALUATION_COLUMNS,
    MODEL_GENERATION_COLUMNS
)


class Plotter:

    def __init__(self) -> None:
        """Initializes the Plotter Class."""
        pass

    @staticmethod
    def display_confusion_matrix(
        conf_matrix: np.ndarray,
        title: str,
        labels: List[Any],
        percentages: bool,
        path: str
    ) -> None:
        """Displays the passed Confusion Matrix.

        Args:
            `confusion_matrix` (numpy.ndarray): Confusion Matrix to display.

            `title` (str): Plot title.

            `labels` (List[Any]): Labels for Columns and Indexes.

            `percentages` (bool): Change color based on category percentage.

            `path` (str): Output save path.
        """
        # SOURCE:
        # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
        # https://stackoverflow.com/a/73154009/14319439

        # Convert Confusion Matrix to DataFrame
        conf_matrix = pd.DataFrame(
            data=conf_matrix,
            index=labels,
            columns=labels
        )

        if percentages:

            # Calculate total elements in categories
            conf_matrix['TOTAL'] = conf_matrix.sum(axis=1)
            conf_matrix.loc['TOTAL'] = conf_matrix.sum()

            # Calculate percentages
            df_percentages = conf_matrix.div(conf_matrix.TOTAL, axis=0)

            # Remove helping columns
            conf_matrix = (
                conf_matrix.drop('TOTAL', axis=1).drop(
                    conf_matrix.tail(1).index
                )
            )
            df_percentages = (
                df_percentages.drop('TOTAL', axis=1).drop(
                    df_percentages.tail(1).index
                )
            )

        # Create new Figure
        figure = plt.figure(figsize=(16, 9))

        # Add simple axes to plot on
        ax = figure.add_subplot(1, 1, 1)

        # Plot the Confusion Matrix
        sns.heatmap(
            data=df_percentages,
            annot=conf_matrix,
            fmt='d',
            ax=ax,
            cbar_kws={'label': 'Percentages'}
        ) if percentages else sns.heatmap(
            data=conf_matrix,
            annot=True,
            fmt='g',
            ax=ax
        )

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
        """Draws line-plot of given data.

        Args:
            `data` (pd.DataFrame): Data to display.

            `columns` (List[str]): Subset of columns to display.

            `title` (str): Plot title.

            `path` (str): Plot output path.
        """
        # Create Plot
        plt.figure(figsize=(16, 9))

        # Draw Plot
        data[columns].plot()

        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Value')

        plt.draw()
        plt.savefig(path)

    def display_training_history(
        self,
        model_name: str,
        history: pd.DataFrame,
        index_col: str,
        path: str,
        timestamp: str
    ) -> None:
        """Displays Training History Plots.

        Args:
            `model_name` (str): Name of NER Model.

            `history` (pandas.DataFrame) : Training History Data.

            `index_col` (str) : Training History Data index column name.

            `path` (str): Training History plot output folder path.

            `timestamp` (str): Training History timestamp for output file name.
        """
        self.__draw_data_plot(
            data=history.set_index(index_col),
            columns=['loss', 'eval_loss'],
            title=f'{model_name} Training History - Loss',
            path=os.path.join(
                path,
                f'training_history_loss_{timestamp}.png'
            )
        )

    def _display_version_data(
        self,
        data: pd.DataFrame,
        parent_version: str,
        output_path: str
    ) -> None:
        """Processes group of ML training runs results.

        Args:
            `data` (pd.DataFrame): ML Generation Runs results data.
            
            `parent_version` (str): ML Generation parent version.
            
            `output_path` (str): Plot output path.
        """
        cols = MODEL_GENERATION_COLUMNS

        if parent_version == 'best':
            cols.remove('parent_version')

        version_data = data[cols]

        if len(version_data) > 10:
            version_data = version_data.sort_values(
                by=['test/macro_f1']
            )[:10]

            name_order = CategoricalDtype(
                natsorted(version_data['Name'].values),
                ordered=True
            )

            version_data['Name'] = version_data['Name'].astype(name_order)

            version_data = version_data.sort_values(by=['Name'])

        for category in EVALUATION_COLUMNS.keys():
            selected_columns = EVALUATION_COLUMNS[category]

            sorted_version_data = version_data.sort_values(
                by=selected_columns,
                ascending=False
            )

            title = (
                'Best SlovakBERT NER Models' if parent_version == 'best' else
                sorted_version_data['parent_version'].values[0]
            )

            # Create a subplot with multiple axes
            fig = sp.make_subplots(rows=1, cols=1)

            # Create bar trace for each column, grouped by 'Name'
            for name in sorted_version_data['Name']:
                x = (
                    sorted_version_data[version_data['Name'] == name]
                    [selected_columns].values[0]
                )

                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=selected_columns,
                        name=name,
                        orientation='h',
                        text=x,
                        textposition='auto'
                    ),
                    col=1,
                    row=1
                )

            # Set the title and axis labels
            fig.update_layout(
                # Plot title
                title=dict(
                    text=(
                        f'<b>{title} {category.capitalize()} Metrics</b>'
                    ),
                    font=dict(size=30),
                    x=0.5
                ),
                xaxis=dict(
                    title='<b>Metric Value</b>',
                    tickfont_size=20,
                    tickprefix="<b>",
                    ticksuffix="</b>"
                ),
                yaxis=dict(
                    title='<b>Metric Name</b>',
                    tickfont_size=20,
                    tickprefix="<b>",
                    ticksuffix="</b>       <br>"
                ),
                legend=dict(
                    traceorder='normal',
                    orientation='h',
                ),
                font=dict(size=25)
            )

            fig.write_html(
                os.path.join(
                    output_path, 'html', f'{parent_version}_{category}.html'
                )
            )

    def process_run_data(
        self,
        run_data_path: str,
        output_path: str,
        separate_generations: bool = True
    ) -> None:
        """Processes ML model training runs results.

        Args:
            `run_data_path` (str): CSV File of ML model training runs results.

            `output_path` (str): Plot output path.

            `separate_generations` (bool, optional): Group generations together
            based on parent version. If it is false it compares best models of
            each generation. In this case provide the bes runs CSV file. 
            Defaults to True.
        """
        data = pd.read_csv(run_data_path)

        if separate_generations:
            data['parent_version'] = data['Name'].str.extract(
                r'(SlovakBERT_NER_Model_V[0-9\.]{5})',
                expand=False
            )
            
            data['parent_version'].fillna(
                'SlovakBERT_NER_Model_V_POC', 
                inplace=True
            )

            for parent_version in sorted(data['parent_version'].unique()):
                self._display_version_data(
                    data=data[data['parent_version'] == parent_version],
                    parent_version=parent_version,
                    output_path=output_path
                )
        else:
            self._display_version_data(
                data=data,
                parent_version='best',
                output_path=output_path
            )
