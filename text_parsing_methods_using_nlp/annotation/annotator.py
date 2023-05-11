import csv
import itertools
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import pipeline

from text_parsing_methods_using_nlp.config.config import (
    ANNOTATED_DATA_FOLDER,
    ANNOTATED_DATA_FOLDER_SLOVAKBERT_NER_VERSION,
    ANNOTATION_PROCESS_OUTPUT_FOLDER,
    CLASSIFICATION_REPORTS_OUTPUT_FOLDER,
    CLASSIFICATION_REPORTS_OUTPUT_FOLDER_SLOVAKBERT_NER_VERSION,
    CONFUSION_MATRICES_OUTPUT_FOLDER,
    CONFUSION_MATRICES_OUTPUT_FOLDER_SLOVAKBERT_NER_VERSION,
    DATA_CONFIG,
    INVERTED_NER_LABELS,
    MANUAL_CORRECTION_DATA_FOLDER,
    METRICS_EVALUATION_OUTPUT_FOLDER,
    METRICS_EVALUATION_OUTPUT_FOLDER_SLOVAKBERT_NER_VERSION,
    MONEY_LEMMAS,
    MONTH_LEMMAS,
    NER_LABELS,
    PERCENTAGE_LEMMAS,
    PREPROCESSING_REGEX_RULES,
    YEAR_PREFIX_LEMMAS,
)
from text_parsing_methods_using_nlp.ops.plotter import Plotter
from text_parsing_methods_using_nlp.utils.utils import setup_folders


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
        model_name: str = 'Crabz_-_SlovakBERT_NER_Model',
        tokenizer: Any = 'crabz/slovakbert-ner',
        timestamp: str = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')
    ) -> None:
        """Initializes the Annotator Class.

        Args:
            `manual_correction_filepath` (str, optional): Manual Correction 
            File Path. Defaults to None.

            `input_data_filepath` (str, optional): Input Data File Path. 
            Defaults to 'NBS_sentence.csv'.

            `input_data_filename` (str, optional): Input Data File Name. 
            Defaults to 'NBS_sentence'.

            `dataset_size` (int, optional): Size of dataset to process. 
            Defaults to NBS_sentence length - 8445.

            `model_test_dataset_evaluation` (bool, optional): Determines if the 
            annotator is used for model performance evaluation on Test dataset.
            Defaults to False.

            `model` (Any, optional): NER Model used for prediction / 
            evaluation. Defaults to 'crabz/slovakbert-ner'.

            `model_name` (str, optional): NER Model name. Defaults to 
            'Crabz_-_SlovakBERT_NER_Model'.

            `tokenizer` (Any, optional): Tokenizer used in pipeline. Defaults 
            to 'crabz/slovakbert-ner'.

            `timestamp` (str, optional): Timestamp for output files. Defaults 
            to current date time timestamp formatted in '%d_%m_%Y__%H_%M_%S'
            format.
        """
        # region Model and Pipeline

        self._model_name = model_name

        self._model_version_present = self._model_name[-1].isdigit()

        self._parent_model = '.'.join(
            self._model_name.split('.')[:-1]
        )

        self._model_test_dataset_evaluation = model_test_dataset_evaluation

        self._ner_pipeline = pipeline(
            task='ner',
            model=model,
            tokenizer=tokenizer
        )

        # endregion

        # region Data

        self._input_data_filename = input_data_filename

        self._data = pd.read_csv(
            input_data_filepath, delimiter=',', encoding='utf-8'
        )

        self._dataset_size = dataset_size

        if self._dataset_size is not None:
            self._data = self._data[:self._dataset_size]
        else:
            self._dataset_size = len(self._data)

        # endregion

        # region File Paths

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
                f'{self._output_file_prefix}_Process_{timestamp}.txt'
            )
        )

        self._classification_report_output_folder_path = os.path.join(
            (
                os.path.join(
                    CLASSIFICATION_REPORTS_OUTPUT_FOLDER_SLOVAKBERT_NER_VERSION,
                    self._parent_model
                ) if self._model_version_present else
                CLASSIFICATION_REPORTS_OUTPUT_FOLDER
            ),
            self._model_name,
        )

        self._classification_report_output_path = os.path.join(
            self._classification_report_output_folder_path,
            f'{self._output_file_prefix}_class_report_{timestamp}.txt'
        )

        self._metrics_evaluation_output_folder_path = os.path.join(
            (
                os.path.join(
                    METRICS_EVALUATION_OUTPUT_FOLDER_SLOVAKBERT_NER_VERSION,
                    self._parent_model
                ) if self._model_version_present else
                METRICS_EVALUATION_OUTPUT_FOLDER
            ),
            self._model_name,
        )

        self._metrics_evaluation_output_path = os.path.join(
            self._metrics_evaluation_output_folder_path,
            f'{self._output_file_prefix}_metrics_evaluation_{timestamp}'
            '.csv'
        )

        self._confusion_matrix_output_folder_path = os.path.join(
            (
                os.path.join(
                    CONFUSION_MATRICES_OUTPUT_FOLDER_SLOVAKBERT_NER_VERSION,
                    self._parent_model
                ) if self._model_version_present else
                CONFUSION_MATRICES_OUTPUT_FOLDER
            ),
            self._model_name,
        )

        self._confusion_matrix_output_path = os.path.join(
            self._confusion_matrix_output_folder_path,
            f'{self._model_name}_{self._dataset_size}_conf_matrix_'
            f'{timestamp}.png'
        )

        self._output_folder_path = os.path.join(
            (
                os.path.join(
                    ANNOTATED_DATA_FOLDER_SLOVAKBERT_NER_VERSION,
                    self._parent_model
                ) if self._model_version_present
                else ANNOTATED_DATA_FOLDER
            ),
            self._model_name,
        )

        self._output_path = os.path.join(
            self._output_folder_path,
            f'{self._model_name}_{self._output_file_prefix}_Annotated_'
            f'{timestamp}.csv'
        )

        setup_folders(
            folders=[
                self._classification_report_output_folder_path,
                self._metrics_evaluation_output_folder_path,
                self._confusion_matrix_output_folder_path,
                self._output_folder_path
            ]
        )

        # endregion

        # region Utils

        self._plotter = Plotter()

        # endregion

    # region Data Manipulation

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

    def _save_data(self) -> None:
        """Saves annotated dataset."""
        self._data.to_csv(
            self._output_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )

    # endregion

    # region Automatic NER Tag Correction

    def _select_tokens(
        self,
        row_index: int,
        word_index: int
    ) -> Tuple[str, str, str, str]:
        """Selects tokens for automated annotation fix operations.

        Args:
            `row_index` (int): Current DataFrame row index.

            `word_index` (int): Current word index.

        Returns:
            Tuple[str, str, str, str]: Collection of contextual words and 
            lemmas.
        """
        next_word_lemma = previous_word_lemma = ''

        text_tokens = self._data['text'][row_index].split()
        lemma_text_tokens = self._data['lemma_text'][row_index].split()

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
            `ner_tags` (List[int]): Current NER Tags generated by model.

            `previous_word_lemma` (str): Lemma of Previous Word.

            `next_word_lemma` (str): Lemma of Next Word.

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
        row_index: int,
        word_index: int
    ) -> List[int]:
        """Corrects NER Tag automatically based on rules. 

        Args:
            `ner_tags` (List[int]): Current token NER Tag.

            `row_index` (int): Current DataFrame row index.

            `word_index` (int): Current token index.

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
        ) = self._select_tokens(row_index, word_index)

        # Fix common miss-matches
        if current_word in ['NBS', 'NAKA'] and ner_tags[0] not in [3, 4]:
            new_ner_tags.append(3)

        # Process Time
        elif re.compile(r'\d{2}:\d{2}').match(current_word):
            new_ner_tags.append(9)

        # Process Number
        elif re.compile(r'[+-]?\d+\.\d+|[+-]?\d').match(current_word):
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

    # endregion

    # region Manual NER Tag Correction

    def _replace_ner_tag(
        self,
        row_id: int,
        token_index: int,
        new_tag: int
    ) -> None:
        """Corrects specific NER Tag based on manual correction info.

        Args:
            `row_id` (int): Miss-matched NER Tag Row ID.

            `token_index` (int): Miss-matched NER Tag Token Index.

            `new_tag` (int): Replacement value.
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

    # endregion

    # region Annotation

    def _apply_annotation(
        self,
        row_index: int,
        classifications: List[Dict[str, Any]],
    ) -> Tuple[List[int], List[int]]:
        """Applies annotation to given row based on Model classifications.

        Args:
            `row_index` (int): Current DataFrame row index.

            `classifications` (List[Dict[str, Any]]): Model classifications for
            given DataFrame row.

        Returns:
            Tuple[List[int], List[int]]: Model NER Output and Corrected NER 
            Output.
        """
        word_index = 0
        current_ner_tag = []
        ner_tags = []
        fixed_ner_tags = []

        for index, classification in enumerate(classifications):
            if self._model_name == 'Crabz_-_SlovakBERT_NER_Model':
                current_ner_tag.append(classification['entity'])
            else:
                current_ner_tag.append(
                    INVERTED_NER_LABELS[classification['entity']]
                )

            current_ner_tag = [current_ner_tag[0]]

            if (
                index < len(classifications) - 1 and
                not classifications[index + 1]['word'].startswith('Ġ')
            ):
                continue

            # Save model prediction
            ner_tags.extend(current_ner_tag)

            if (
                not self._model_test_dataset_evaluation and
                self._model_name == 'Crabz_-_SlovakBERT_NER_Model'
            ):
                # Correct model prediction if needed
                current_ner_tag = self._fix_ner_tags(
                    ner_tags=current_ner_tag,
                    row_index=row_index,
                    word_index=word_index
                )

            # Save corrected values
            fixed_ner_tags.extend(current_ner_tag)

            if not self._manual_correction_file_exists:
                self._document_annotation_process(
                    row_index=row_index,
                    word_index=word_index,
                    current_ner_tag=current_ner_tag
                )

            word_index += 1

            current_ner_tag = []

        return ner_tags, fixed_ner_tags

    def _annotate(self) -> None:
        """Annotates input dataset."""
        if not self._model_test_dataset_evaluation:
            self._preprocess_data()

        ner_tags_col = []
        fixed_ner_tags_col = []

        for i in tqdm(range(self._dataset_size)):
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

            ner_tags, fixed_ner_tags = self._apply_annotation(
                row_index=i,
                classifications=classifications
            )

            ner_tags_col.append(ner_tags)

            if not self._model_test_dataset_evaluation:
                fixed_ner_tags_col.append(fixed_ner_tags)

        self._data['model_ner_tags'] = ner_tags_col

        if not self._model_test_dataset_evaluation:
            self._data['fixed_ner_tags'] = fixed_ner_tags_col

        if self._manual_correction_file_exists:
            self._manual_correction()

        self._save_data()

    # endregion

    # region Annotation Evaluation

    def _evaluate_using_classification_report(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> None:
        """Evaluates Model Annotation Performance using a Classification 
        Report.

        Args:
            `y_true` (pd.Series): True values.

            `y_pred` (pd.Series): Predicted values.
        """
        with open(self._classification_report_output_path, 'w+') as output:
            output.write(classification_report(y_true, y_pred))

    def _evaluate_using_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> None:
        """Evaluates Model Annotation Performance using a Standardized Metrics.

        Args:
            `y_true` (pd.Series): True values.

            `y_pred` (pd.Series): Predicted values.
        """
        pd.DataFrame(
            {
                'Name': [self._model_name],
                'test/precision': [
                    precision_score(y_true, y_pred, average='weighted')
                ],
                'test/macro_precision': [
                    precision_score(y_true, y_pred, average='macro')
                ],
                'test/recall': [
                    recall_score(y_true, y_pred, average='weighted')
                ],
                'test/macro_recall': [
                    recall_score(y_true, y_pred, average='macro')
                ],
                'test/f1': [
                    f1_score(y_true, y_pred, average='weighted')
                ],
                'test/macro_f1': [
                    f1_score(y_true, y_pred, average='macro')
                ],
                'test/accuracy': [accuracy_score(y_true, y_pred)]
            }
        ).to_csv(self._metrics_evaluation_output_path, index=False)

    def _process_columns_for_evaluation(
        self,
        true_column: str
    ) -> Tuple[pd.Series, pd.Series]:
        """Processes loaded dataset columns to True and Predicted values before
        evaluation process.

        Args:
            `true_column` (str): True column name.

        Returns:
            Tuple[pd.Series, pd.Series]: Processed True and Predicted values.
        """
        self._data['model_ner_tags'] = self._data['model_ner_tags'].apply(
            eval
        )

        if self._model_test_dataset_evaluation:
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
            self._data[true_column] = self._data[true_column].apply(eval)

        # Merge DataFrame rows together
        return (
            pd.Series(
                list(itertools.chain(*self._data[true_column].tolist()))
            ),
            pd.Series(
                list(itertools.chain(*self._data['model_ner_tags'].tolist()))
            )
        )

    def _evaluate(self) -> None:
        """Evaluates Model Performance based on predicted and corrected 
        values."""
        self._data = pd.read_csv(self._output_path)

        # Generate y_true, y_pred from dataset
        y_true, y_pred = self._process_columns_for_evaluation(
            true_column=(
                'ner_tags' if self._model_test_dataset_evaluation else
                'fixed_ner_tags'
            )
        )

        # Evaluate Model Performance using Classification Report and
        # Standardized Metrics
        # self._evaluate_using_classification_report(y_true, y_pred)
        self._evaluate_using_metrics(y_true, y_pred)

        # Generate Confusion Matrix
        self._plotter.display_confusion_matrix(
            conf_matrix=confusion_matrix(y_true, y_pred),
            title=(
                f'{self._model_name} {self._input_data_filename} '
                f'{self._dataset_size} Confusion Matrix'
            ),
            labels=INVERTED_NER_LABELS.keys(),
            percentages=True,
            path=self._confusion_matrix_output_path
        )

    # endregion

    # region Utils

    def _document_annotation_process(
        self,
        row_index: int,
        word_index: int,
        current_ner_tag: List[int]
    ) -> None:
        """Documents annotation process.

            This will allows us to create a manual correction file.

        Args:
            `row_index` (int): Current processed row index.

            `word_index` (int): Current processed word index.

            `current_ner_tag` (List[int]): Current word NER Tag.
        """
        ner_tag = (
            current_ner_tag[0] if isinstance(
                current_ner_tag[0], str
            ) else NER_LABELS[current_ner_tag[0]]
        )

        process_messages = [
            f"Row: {row_index + 1}\n",
            f"ID: {self._data['id'][row_index]}\n",
            f'Token Index: {word_index}\n',
            f"Word: {self._data['text_tokens'][row_index][word_index]}\n",
            f'NER Tag: {ner_tag} -- \n\n'
        ]

        with open(self._annotation_process_output_path, 'a+') as process_file:
            for process_message in process_messages:
                process_file.write(process_message)

    def __call__(self) -> None:
        """Makes Annotator Class callable."""
        self._annotate()

        if (
            self._manual_correction_file_exists or
            self._model_test_dataset_evaluation
        ):
            self._evaluate()

    # endregion
