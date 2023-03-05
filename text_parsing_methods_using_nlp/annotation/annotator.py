import csv
import itertools
import os
import re
from datetime import datetime
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline


class Annotator:

    """Represent the Text Annotator."""

    def __init__(
        self,
        manual_correction_filepath: str,
        input_data_filepath: str = os.path.join(
            os.getcwd(),
            'text_parsing_methods_using_nlp',
            'data',
            'NBS_sentence.csv'
        ),
        dataset_size: int = 100
    ) -> None:
        """Initializes the Annotator Class.

        Args:
            manual_correction_filepath (str): Manual Correction File Path.
            input_data_filepath (str, optional): Input Data File Path. Defaults
            to 'Project_root...NBS_sentence.csv'.
            dataset_size (int, optional): Size of dataset to process. Defaults
            to 100.
        """
        self._timestamp = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')
        self._data = pd.read_csv(
            input_data_filepath, delimiter=',', encoding='utf-8'
        )[:dataset_size]

        self._manual_correction_filepath = manual_correction_filepath

        self._output_path = (
            f"{input_data_filepath.split('.')[0]}_annotated_"
            f"{self._timestamp}.csv"
        )

        self._ner_pipeline = pipeline(
            task='ner',
            model='crabz/slovakbert-ner'
        )

        self._dataset_size = dataset_size

        self._ner_labels = {
            0: '0',
            1: 'B-Person',
            2: 'I-Person',
            3: 'B-Organization',
            4: 'I-Organization',
            5: 'B-Location',
            6: 'I-Location',
            7: 'B-Date',
            8: 'I-Date',
            9: 'Time',
            10: 'B-Money',
            11: 'I-Money',
            12: 'B-Percentage',
            13: 'I-Percentage',
        }

        self._ner_labels_inverted = {v: k for k, v in self._ner_labels.items()}

        self._month_lemmas = [
            'január',
            'február',
            'marec',
            'apríl',
            'máj',
            'jún',
            'júl',
            'august',
            'september',
            'október',
            'november',
            'december'
        ]

        self._year_prefix_lemmas = ['', 'štvrťrok', 'polrok', 'rok']

        self._money_lemmas = ['tisíc', 'mil', 'milión', 'miliarda', 'euro']

        self._percentage_lemmas = ['%', 'percento', 'p.b.']

    def _preprocess_data(self) -> None:
        """Pre-processes dataset."""
        regex_mapping = {

            # Add an extra space before colon which is not a time indicator
            r'([a-zA-Z]+):([a-zA-Z]+)': r'\1 : \2',

            # Remove reference indicators
            r'(\[\s*\d+\s*\])': r' ',

            # Remove special characters
            r'''['"`‘’„“”\(\)\[\]\/(\s\-|–\s))!?;]|(\s*:\s+)|(\.|,)\s*(\.|,)''': r' ',

            # Remove dots from text
            r'([a-zA-Z]{2,})(\s*\.\s*)': r'\1 ',

            # Remove dots from text
            r'([ľščťžýáíéóúäôňďĺ%]{1,}[a-zA-Z0-9]*)\s*(\.)\s*': r'\1 ',

            # Remove dots from text
            r'(\d{1,})(\s*\.\s*)': r'\1 ',

            # Replace floating point number commas with dot notation
            r'([+-]?\d+),(\d+)': r'\1.\2',

            # Add extra space to floating point percentages
            r'([+-]?\d+\.\d+|[+-]?\d)(%)': r'\1 \2 ',

            # Remove commas from text
            r',': r' ',

            # Replace excessive whitespace characters
            r'\s+': r' ',

            # Merge larger number formats together
            r'( [+-]?\d{1,3}) (\d+) ([a-z]*)': r'\1\2 \3',

            # Replace Euro symbol
            r'€': 'euro',

            # Remove extra space after letters
            r'(\s+[a-zA-Zľščťžýáíéóúäôňďĺ]) \.': r'\1.',

            # Replace specific percentage value
            r'p\. b\.': 'p.b.',

            # Replace specific monetary value
            r'desaťtisíc': '10 tisíc'
        }

        # Apply regex replacements to dataset
        for to_replace, replacement in regex_mapping.items():
            self._data.replace(
                to_replace, replacement, regex=True, inplace=True
            )

        # Remove excessive leading and tailing whitespaces
        self._data = self._data.apply(
            lambda x: x.str.strip() if x.dtype == "object" else x
        )

        # Tokenize text and lemmas
        for col in ['text', 'lemma_text']:
            self._data[f'{col}_tokens'] = self._data[col].str.split(' ')

    def _save_data(self) -> None:
        """Saves annotated dataset."""
        self._data.to_csv(
            self._output_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )

    # Automatic NER Tag Correction

    def _select_tokens(
        self,
        row: int,
        word_index: int
    ) -> Tuple[str, str, str, str]:
        """Selects tokens for automated annotation fix operations.

        Args:
            row (int): Current DataFrame row.
            word_index (int): Current word index.

        Returns:
            Tuple[str, str, str, str]: Collection of contextual words and 
            lemmas.
        """
        next_word_lemma = previous_word_lemma = ''

        text_tokens = self._data['text'][row].split()
        lemma_text_tokens = self._data['lemma_text'][row].split()

        # Get Previous Lemma
        if word_index > 1:
            previous_word_lemma = lemma_text_tokens[word_index - 1]

        # Get Current Word and Lemma
        current_word = text_tokens[word_index]
        current_word_lemma = lemma_text_tokens[word_index]

        # Get Next Lemma
        if word_index < len(text_tokens) - 1:
            next_word_lemma = lemma_text_tokens[word_index + 1]

        return (
            previous_word_lemma,
            current_word,
            current_word_lemma,
            next_word_lemma
        )

    def _process_numbers(
        self,
        ner_tags: List[int],
        previous_word_lemma: str,
        next_word_lemma: str
    ) -> int:
        """Processes number tokens.

        Args:
            ner_tags (List[int]): Current NER Tags generated by model.
            previous_word_lemma (str): Lemma of Previous Word.
            next_word_lemma (str): Lemma of Next Word.

        Returns:
            int: New NER Tag for current token.
        """
        new_ner_tag = 0

        if (
            next_word_lemma in self._month_lemmas or
            previous_word_lemma in self._year_prefix_lemmas
        ):
            new_ner_tag = 7
        elif previous_word_lemma in self._month_lemmas or ner_tags[-1] == 7:
            new_ner_tag = 8
        elif next_word_lemma in self._money_lemmas:
            new_ner_tag = 10
        elif next_word_lemma in self._percentage_lemmas:
            new_ner_tag = 12

        return new_ner_tag

    def _fix_ner_tags(
        self,
        ner_tags: List[int],
        row: int,
        word_index: int
    ) -> List[int]:
        """Corrects NER Tag automatically based on rules. 

        Args:
            ner_tags (List[int]): Current token NER Tag.
            row (int): Current DataFrame row.
            word_index (int): Current token index.

        Returns:
            List[int]: New NER Tag for current token.
        """
        new_ner_tags = []

        previous_word_lemma, current_word, current_word_lemma, next_word_lemma = self._select_tokens(
            row, word_index
        )

        # Fix common miss-matches
        if current_word in ['NBS', 'NAKA'] and ner_tags[0] not in [3, 4]:
            new_ner_tags.append(3)

        # Process Time
        elif re.compile(r'\d{2}:\d{2}').match(current_word):
            new_ner_tags.append(9)

        # Process Number
        elif re.compile(r'[+-]?\d+\.\d+|[+-]?\d').match(current_word):
            # # Old Version
            # if (
            #     next_word_lemma in self._month_lemmas or
            #     previous_word_lemma in self._year_prefix_lemmas
            # ):
            #     new_ner_tags.append(7)
            # elif previous_word_lemma in self._month_lemmas or ner_tags[-1] == 7:
            #     new_ner_tags.append(8)
            # elif next_word_lemma in self._money_lemmas:
            #     new_ner_tags.append(10)
            # elif next_word_lemma in self._percentage_lemmas:
            #     new_ner_tags.append(12)
            # else:
            #     new_ner_tags.append(0)

            # Refactored Version
            new_ner_tags.append(
                self._process_numbers(
                    ner_tags,
                    previous_word_lemma,
                    next_word_lemma
                )
            )

        # Process Date
        elif current_word_lemma in self._month_lemmas:
            if previous_word_lemma.isdigit():
                new_ner_tags.append(8)
            else:
                new_ner_tags.append(7)

        # Process Money
        elif current_word_lemma in self._money_lemmas:
            new_ner_tags.append(11)

        # Process Percentage
        elif current_word_lemma in self._percentage_lemmas:
            new_ner_tags.append(13)

        # No fix is needed
        else:
            new_ner_tags = ner_tags

        return new_ner_tags

    # Manual NER Tag Correction

    def _replace_ner_tag(
        self,
        row_id: int,
        token_index: int,
        new_tag: int
    ) -> None:
        """Corrects specific NER Tag based on manual correction info.

        Args:
            row_id (int): Miss-matched NER Tag Row ID.
            token_index (int): Miss-matched NER Tag Token Index.
            new_tag (int): Replacement value.
        """
        # Get the row with the specified ID
        row = self._data[self._data['id'] == row_id].iloc[0]

        # Get the list of fixed_ner_tags
        fixed_ner_tags = row['fixed_ner_tags']

        # Overwrite the value at the specified list index
        fixed_ner_tags[token_index] = new_tag

        # Update the DataFrame with the new fixed_ner_tags value
        self._data[
            self._data['id'] == row_id
        ]['fixed_ner_tags'] = str(fixed_ner_tags)

    def _manual_correction(self) -> None:
        """Runs NER Tag correction based on Manual correction file."""
        with open(self._manual_correction_filepath, 'r') as f:
            for section in [
                s.strip().split('\n') for s in f.read().split('\n\n')
            ]:
                self._replace_ner_tag(
                    row_id=int(section[0].split()[-1]),
                    token_index=int(section[1].split()[-1]),
                    new_tag=self._ner_labels_inverted[
                        section[-1].split('--')[-1].strip()
                    ]
                )

    def _annotate(self) -> None:
        """Annotates input dataset."""
        self._preprocess_data()

        ner_tags_col = []
        fixed_ner_tags_col = []

        for i in range(self._dataset_size):
            classifications = self._ner_pipeline(self._data['text'][i])

            word_index = 0
            ner_tags = []
            fixed_ner_tags = []
            current_ner_tag = []

            for index, classification in enumerate(classifications):
                current_ner_tag.append(classification['entity'])

                current_ner_tag = [current_ner_tag[0]]

                if (
                    index < len(classifications) - 1 and
                    not classifications[index + 1]['word'].startswith(
                        tuple(['Ġ', ','])
                    )
                ):
                    continue
                    
                # Save model prediction
                ner_tags.extend(current_ner_tag)

                # Correct model prediction if needed
                current_ner_tag = self._fix_ner_tags(
                    ner_tags=current_ner_tag,
                    row=i,
                    word_index=word_index
                )

                # Save corrected values
                fixed_ner_tags.extend(current_ner_tag)

                # # For Manual Annotation purposes
                # print(f"ID: {self._data['id'][i]}")
                # print(f"Token Index: {word_index}")
                # print(f"Word: {self._data['text_tokens'][i][word_index]}")
                # print(
                #     f"NER Tag: {self._ner_labels[current_ner_tag[0]]}",
                #     end='\n\n'
                # )

                word_index += 1

                current_ner_tag = []

            ner_tags_col.append(ner_tags)
            fixed_ner_tags_col.append(fixed_ner_tags)

        self._data['ner_tags'] = ner_tags_col
        self._data['fixed_ner_tags'] = fixed_ner_tags_col

        self._manual_correction()

        self._save_data()

    @staticmethod
    def display_confusion_matrix(
            conf_matrix: np.ndarray,
            labels: List[Any],
            path: str
    ) -> None:
        """Displays the passed Confusion Matrix.
        Args:
            confusion_matrix (numpy.ndarray): Confusion Matrix to display
            labels (List[Any]): Labels for Columns and Indexes
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

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')

        plt.draw()
        plt.savefig(path)
        plt.show(block=False)

    def _evaluate(self) -> None:
        """Evaluates Model Performance based on predicted and corrected 
        values."""
        self._data = pd.read_csv(self._output_path)

        # Turn String cols to List[int] values
        self._data['ner_tags'] = self._data['ner_tags'].apply(eval)
        self._data['fixed_ner_tags'] = self._data['fixed_ner_tags'].apply(eval)

        # Merge DataFrame rows together
        y_true = pd.Series(
            list(itertools.chain(*self._data['fixed_ner_tags'].tolist()))
        )
        y_pred = pd.Series(
            list(itertools.chain(*self._data['ner_tags'].tolist()))
        )
        
        # Evaluate Model Performance
        print(classification_report(y_true, y_pred))        

        # Generate Confusion Matrix
        self.display_confusion_matrix(
            conf_matrix = confusion_matrix(y_true, y_pred),
            labels = self._ner_labels_inverted.keys(),
            path = os.path.join(
                os.getcwd(), 
                'output',
                'plots', 
                'confusion_matrices',
                f'conf_matrix_{self._timestamp}.png'
            )
        )

    def __call__(self) -> None:
        """Makes Annotator Class callable."""
        self._annotate()
        self._evaluate()
