import csv
import os
import string
from pprint import pprint

import pandas as pd
from transformers import pipeline


class Annotator:

    def __init__(
        self,
        input_data_filepath: str = os.path.join(
            os.getcwd(),
            'text_parsing_methods_using_nlp',
            'data',
            'NBS_sentence.csv'
        )
    ) -> None:
        # TODO - Docstring
        self._data = pd.read_csv(input_data_filepath, delimiter=',')
        self._ner_pipeline = pipeline(
            task='ner',
            # model='crabz/slovakbert-ner'
            model='crabz/fernet-cc_sk-ner'

        )
        self._output_path = (
            f"{input_data_filepath.split('.')[0]}_annotated.csv"
        )

    def _preprocess_data(self) -> None:
        # TODO - Docstring

        regex_mapping = {
            # Replace floating point number commas with dot notation
            r'([+-]?[0-9]+),([0-9]+)': r'\1.\2',

            # Add extra space to floating point percentages
            r'([+-]?[0-9]+.[0-9]+|[+-]?[0-9])(%)': r'\1 \2',

            # Add an extra space before colon which is not a time indicator
            r'([a-zA-Z]+):([a-zA-Z]+)': r'\1 : \2',

            # Remove special characters
            r'''['"`„“\(\)\/(\s\-|–\s))]|(: )''': r' ',  # add , if needed

            # Remove dots from text
            r'([a-zA-Z]{2,})(\.\s+)([a-zA-Z]+)': r'\1 \3',

            # Replace excessive whitespace characters
            r'\s+': r' ',

            # Merge larger number formats together
            r'( [+-]?[0-9]{1,3}) ([0-9]+) ([a-z]+)': r'\1\2 \3'
        }

        for to_replace, replacement in regex_mapping.items():
            self._data.replace(
                to_replace, replacement, regex=True, inplace=True
            )

        # Remove excessive leading and tailing whitespaces
        self._data = self._data.apply(
            lambda x: x.str.strip() if x.dtype == "object" else x
        )

    def _save_data(self) -> None:
        # TODO - Docstring
        self._data.to_csv(
            self._output_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )

    def annotate(self) -> None:
        # TODO - Docstring

        # TODO - Refactor

        self._preprocess_data()
        self._save_data()

        for i in range(len(self._data)):
            classifications = self._ner_pipeline(self._data['text'][i])

            tokens = self._data['lemma_text'][i].split()
            words = []
            ner_tags = []
            current_words = []
            current_ner_tag = []

            for index, classification in enumerate(classifications):
                current_words.append(classification['word'].replace('#', ''))

                current_ner_tag.append(classification['entity'])

                current_words = [''.join(current_words)]

                if len(set(current_ner_tag)) == 1:
                    current_ner_tag = [current_ner_tag[0]]

                if (
                    index < len(classifications) - 1 and (
                        classifications[index + 1]['word'].startswith('#')
                    )
                ):
                    continue

                if (
                    index < len(classifications) - 2
                    and (
                        (
                            (classification['word'].isdigit()
                             or len(classification['word']) == 1)
                            and classifications[index + 1]['word'].startswith(
                                tuple(['.', ':'])  # comma
                            )
                        )
                        or (
                            classification['word'].startswith(
                                tuple(['.', ':'])  # comma
                            )
                            and classifications[index + 1]['word'].isdigit()
                        )
                    )
                ):
                    continue

                print(current_words)
                print(current_ner_tag)

                ner_tags.extend(current_ner_tag) if len(
                    current_ner_tag) == 1 else ner_tags.append(current_ner_tag)
                words.extend(current_words)

                current_words = []
                current_ner_tag = []

            print(f'Our: {words}')
            print(f'Official: {tokens}')
            print(ner_tags)

            print(f'Same Length: {len(tokens) == len(ner_tags)}')
            print(len(tokens))
            print(len(ner_tags), end='\n\n')

            # self._save_data()
