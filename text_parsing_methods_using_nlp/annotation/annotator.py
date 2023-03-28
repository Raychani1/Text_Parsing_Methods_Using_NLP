import csv
import itertools
import os
import re
from datetime import datetime
from typing import Any, List, Tuple

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

from text_parsing_methods_using_nlp.config import (
    ANNOTATED_DATA_FOLDER,
    ANNOTATION_PROCESS_OUTPUT_FOLDER,
    CLASSIFICATION_REPORTS_OUTPUT_FOLDER,
    CONFUSION_MATRICES_OUTPUT_FOLDER,
    DATA_CONFIG,
    INVERTED_NER_LABELS,
    MANUAL_CORRECTION_DATA_FOLDER,
    MONEY_LEMMAS,
    MONTH_LEMMAS,
    NER_LABELS,
    PERCENTAGE_LEMMAS,
    PREPROCESSING_REGEX_RULES,
    YEAR_PREFIX_LEMMAS,
)
from text_parsing_methods_using_nlp.ops.plotter import Plotter

# Set data config
config = DATA_CONFIG['NBS_sentence']


class Annotator:

    """Represent the Text Annotator."""

    def __init__(
        self,
        manual_correction_filepath: str = None,
        input_data_filepath: str = config['raw_input_data_path'],
        input_data_filename: str = config['file_name'],
        dataset_size: int = config['dataset_size'],
        model_test_dataset_evaluation: bool = False,
        model: Any = 'crabz/slovakbert-ner',
        tokenizer: Any = 'crabz/slovakbert-ner',
        timestamp: str = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')
    ) -> None:
        """Initializes the Annotator Class.

        Args:
            manual_correction_filepath (str, optional): Manual Correction File
            Path. Defaults to None.
            input_data_filepath (str, optional): Input Data File Path. Defaults
            to 'NBS_sentence.csv'.
            input_data_filename (str, optional): Input Data File Name. Defaults
            to 'NBS_sentence'.
            dataset_size (int, optional): Size of dataset to process. Defaults
            to NBS_sentence length - 8445.
            model_test_dataset_evaluation (bool, optional): Determines if the
            annotator is used for model performance evaluation on Test dataset.
            Defaults to False.
            model (Any, optional): NER Model used for prediction / evaluation.
            Defaults to 'crabz/slovakbert-ner'.
            tkenizer (Any, optional): Tokenizer used in pipeline. Defaults to 
            'crabz/slovakbert-ner'.
            timestamp (str, optional): Timestamp for output files. Defaults to 
            current date time timestamp formatted in '%d_%m_%Y__%H_%M_%S'
            format.
        """
        self._timestamp = timestamp

        self._dataset_size = dataset_size

        self._model_test_dataset_evaluation = model_test_dataset_evaluation

        self._data = pd.read_csv(
            input_data_filepath, delimiter=',', encoding='utf-8'
        )[:self._dataset_size]

        self._output_file_prefix = (
            input_data_filename if self._model_test_dataset_evaluation else 
            f'{input_data_filename}_{self._dataset_size}' 
        )

        self._manual_correction_filepath = (
            os.path.join(
                MANUAL_CORRECTION_DATA_FOLDER,
                f"{self._output_file_prefix}_Manual.txt"
            ) if manual_correction_filepath is None
            else manual_correction_filepath
        )

        self._manual_correction_file_exists = os.path.exists(
            self._manual_correction_filepath
        )

        self._annotation_process_output_path = (
            None if self._manual_correction_file_exists else os.path.join(
                ANNOTATION_PROCESS_OUTPUT_FOLDER,
                f"{self._output_file_prefix}_Process_{self._timestamp}.txt"
            )
        )

        self._classification_report_output_path = os.path.join(
            CLASSIFICATION_REPORTS_OUTPUT_FOLDER,
            f"{self._output_file_prefix}_class_report_{self._timestamp}.txt"
        )
        
        self._output_path = os.path.join(
            ANNOTATED_DATA_FOLDER,
            f"{self._output_file_prefix}_Annotated_{self._timestamp}.csv"
        )

        self._ner_pipeline = pipeline(
            task='ner',
            model=model,
            tokenizer=tokenizer
        )

        self._plotter = Plotter()

    def _preprocess_data(self) -> None:
        """Pre-processes dataset."""
        # Apply regex replacements to dataset
        for to_replace, replacement in PREPROCESSING_REGEX_RULES.items():
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
    
    def _document_annotation_process(
        self,
        row_index: int,
        word_index: int,
        current_ner_tag: List[int]
    ) -> None:
        """Documents annotation process.
        
            This will allows us to create a manual correction file.

        Args:
            row_index (int): Current processed row index.
            word_index (int): Current processed word index.
            current_ner_tag (List[int]): Current word NER Tag.
        """
        process_messages = [
            f"Row: {row_index + 1}\n",
            f"ID: {self._data['id'][row_index]}\n",
            f'Token Index: {word_index}\n',
            f"Word: {self._data['text_tokens'][row_index][word_index]}\n",
            f'NER Tag: {NER_LABELS[current_ner_tag[0]]} -- \n\n'
        ]

        with open(self._annotation_process_output_path, 'a+') as process_file:
            for process_message in process_messages:
                process_file.write(process_message)

    def _save_data(self) -> None:
        """Saves annotated dataset."""
        self._data.to_csv(
            self._output_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )

    ###########################################################################
    #                      Automatic NER Tag Correction                       #
    ###########################################################################

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
            next_word_lemma in MONTH_LEMMAS or
            previous_word_lemma in YEAR_PREFIX_LEMMAS
        ):
            new_ner_tag = 7
        elif previous_word_lemma in MONTH_LEMMAS or ner_tags[-1] == 7:
            new_ner_tag = 8
        elif next_word_lemma in MONEY_LEMMAS:
            new_ner_tag = 10
        elif next_word_lemma in PERCENTAGE_LEMMAS:
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

        # Select Tokens
        (
            previous_word_lemma, 
            current_word, 
            current_word_lemma, 
            next_word_lemma
        ) = self._select_tokens(row, word_index)

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
            #     next_word_lemma in MONTH_LEMMAS or
            #     previous_word_lemma in YEAR_PREFIX_LEMMAS
            # ):
            #     new_ner_tags.append(7)
            # elif previous_word_lemma in MONTH_LEMMAS or ner_tags[-1] == 7:
            #     new_ner_tags.append(8)
            # elif next_word_lemma in MONEY_LEMMAS:
            #     new_ner_tags.append(10)
            # elif next_word_lemma in PERCENTAGE_LEMMAS:
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
        elif current_word_lemma in MONTH_LEMMAS:
            if previous_word_lemma.isdigit():
                new_ner_tags.append(8)
            else:
                new_ner_tags.append(7)

        # Process Money
        elif current_word_lemma in MONEY_LEMMAS:
            new_ner_tags.append(11)

        # Process Percentage
        elif current_word_lemma in PERCENTAGE_LEMMAS:
            new_ner_tags.append(13)

        # No fix is needed
        else:
            new_ner_tags = ner_tags

        return new_ner_tags

    ###########################################################################
    #                        Manual NER Tag Correction                        #
    ###########################################################################

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
                    row_id=int(section[1].split()[-1]),
                    token_index=int(section[2].split()[-1]),
                    new_tag=INVERTED_NER_LABELS[
                        section[-1].split('--')[-1].strip()
                    ]
                )

    def _annotate(self) -> None:
        """Annotates input dataset."""

        if not self._model_test_dataset_evaluation:
            self._preprocess_data()

        ner_tags_col = []
        fixed_ner_tags_col = []

        for i in range(self._dataset_size):
            
            print(f'Annotation Progress: {i} / {self._dataset_size}')

            classifications = self._ner_pipeline(
                ' '.join(
                    re.sub(
                        r'[\'\s+]+', 
                        ' ',
                        self._data['tokens'][i][1:-1]).strip().split(' ')
                )
                if self._model_test_dataset_evaluation 
                else self._data['text'][i]
            )      

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

                if not self._model_test_dataset_evaluation:

                    # TODO - Extract functionality to reduce complexity

                    # Correct model prediction if needed
                    current_ner_tag = self._fix_ner_tags(
                        ner_tags=current_ner_tag,
                        row=i,
                        word_index=word_index
                    )

                    # Save corrected values
                    fixed_ner_tags.extend(current_ner_tag)

                    if not self._manual_correction_file_exists:
                        self._document_annotation_process(
                            row_index=i,
                            word_index=word_index,
                            current_ner_tag=current_ner_tag
                        )

                word_index += 1

                current_ner_tag = []

            ner_tags_col.append(ner_tags)

            if not self._model_test_dataset_evaluation:
                fixed_ner_tags_col.append(fixed_ner_tags)

        self._data['model_ner_tags'] = ner_tags_col

        if not self._model_test_dataset_evaluation:
            self._data['fixed_ner_tags'] = fixed_ner_tags_col

        if self._manual_correction_file_exists:
            self._manual_correction()

        self._save_data()

    @staticmethod
    def replace(x):
        for what, new in INVERTED_NER_LABELS.items(): # or iteritems in Python 2
            x = x.replace(what, str(new))
        return x

    def _evaluate(self) -> None:
        """Evaluates Model Performance based on predicted and corrected 
        values."""
        self._data = pd.read_csv(self._output_path)

        true_column = (
            'ner_tags' if self._model_test_dataset_evaluation else 
            'fixed_ner_tags'
        )

        if self._model_test_dataset_evaluation:
            self._data['model_ner_tags'] = self._data['model_ner_tags'].apply(
                eval
            ).apply(lambda a: list(map(int, list(map(self.replace, a)))))

            self._data[true_column] = self._data[true_column].apply(
                lambda x: [
                    int(i) for i in re.sub(
                        r'\s+', 
                        ' ', 
                        x[1:-1]
                    ).strip().split(' ')
                ]             
            )
        else:
            # Turn String cols to List[int] values
            self._data['model_ner_tags'] = self._data['model_ner_tags'].apply(
                eval
            )
            self._data[true_column] = self._data[true_column].apply(eval)
        # Merge DataFrame rows together
        y_true = pd.Series(
            list(itertools.chain(*self._data[true_column].tolist()))
        )
        y_pred = pd.Series(
            list(itertools.chain(*self._data['model_ner_tags'].tolist()))
        )

        print('Y True: ', set(list(itertools.chain(*self._data[true_column].tolist()))))
        print('Y Pred: ', set(list(itertools.chain(*self._data['model_ner_tags'].tolist()))))

        # Evaluate Model Performance
        with open(self._classification_report_output_path, 'w+') as output:
            output.write(classification_report(y_true, y_pred))

        print(confusion_matrix(y_true, y_pred))

        # Generate Confusion Matrix
        self._plotter.display_confusion_matrix(
            conf_matrix=confusion_matrix(y_true, y_pred),
            title=f'SlovakBERT NER {self._dataset_size} Confusion Matrix',
            labels=INVERTED_NER_LABELS.keys(),
            path=os.path.join(
                CONFUSION_MATRICES_OUTPUT_FOLDER,
                f'slovakbert_ner_{self._dataset_size}_conf_matrix_'
                f'{self._timestamp}.png'
            )
        )

    def __call__(self) -> None:
        """Makes Annotator Class callable."""
        self._annotate()

        if (
            self._manual_correction_file_exists or 
            self._model_test_dataset_evaluation
        ):
            self._evaluate()
        
