import csv
import os
import re
from typing import List, Tuple

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
        ),
        dataset_size: int = 500
    ) -> None:
        # TODO - Docstring

        self._data = pd.read_csv(
            input_data_filepath, delimiter=',', encoding='utf-8'
        )[:dataset_size]

        self._ner_pipeline = pipeline(
            task='ner',
            model='crabz/slovakbert-ner'
        )

        self._output_path = (
            f"{input_data_filepath.split('.')[0]}_annotated.csv"
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
        # TODO - Docstring

        regex_mapping = {

            # Add an extra space before colon which is not a time indicator
            r'([a-zA-Z]+):([a-zA-Z]+)': r'\1 : \2',

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

            r',': r' ',

            # Replace excessive whitespace characters
            r'\s+': r' ',

            # Merge larger number formats together
            r'( [+-]?\d{1,3}) (\d+) ([a-z]*)': r'\1\2 \3',

            # Replace Euro symbol
            r'€': 'euro',

            # Remove extra space after letters
            r'(\s+[a-zA-Zľščťžýáíéóúäôňďĺ]) \.': r'\1.',

            r'p\. b\.': 'p.b.',

            r'desaťtisíc': '10 tisíc'
        }

        for to_replace, replacement in regex_mapping.items():
            self._data.replace(
                to_replace, replacement, regex=True, inplace=True
            )

        # Remove excessive leading and tailing whitespaces
        self._data = self._data.apply(
            lambda x: x.str.strip() if x.dtype == "object" else x
        )

        for col in ['text', 'lemma_text']:
            self._data[f'{col}_tokens'] = self._data[col].str.split(' ')

    def _save_data(self) -> None:
        # TODO - Docstring
        self._data.to_csv(
            self._output_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )

    def _select_tokens(
        self,
        row: int,
        word_index: int
    ) -> Tuple[str, str, str, str]:
        # TODO - Docstring
        next_word_lemma = previous_word_lemma = ''

        text_tokens = self._data['text'][row].split()
        lemma_text_tokens = self._data['lemma_text'][row].split()

        # Get Previous Word and Lemma
        if word_index > 1:
            previous_word_lemma = lemma_text_tokens[word_index - 1]

        # Get Current Word and Lemma
        current_word = text_tokens[word_index]
        current_word_lemma = lemma_text_tokens[word_index]

        # Get Next Word and Lemma
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
        # TODO - Docstring
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

    def _fix_ner_tags(self, ner_tags: List[int], row: int, word_index: int):
        # TODO - Docstring

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
            # Old Version
            if (
                next_word_lemma in self._month_lemmas or 
                previous_word_lemma in self._year_prefix_lemmas
            ):
                new_ner_tags.append(7)
            elif previous_word_lemma in self._month_lemmas or ner_tags[-1] == 7:
                new_ner_tags.append(8)
            elif next_word_lemma in self._money_lemmas:
                new_ner_tags.append(10)
            elif next_word_lemma in self._percentage_lemmas:
                new_ner_tags.append(12)
            # else:
            #     new_ner_tags.append(0)

            # # Refactored Version
            # new_ner_tags.append(
            #     self._process_numbers(
            #         ner_tags,
            #         previous_word_lemma,
            #         next_word_lemma
            #     )
            # )

        # Process Date
        elif current_word_lemma in self._month_lemmas:
            if ner_tags[-1] == 7:
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

    def annotate(self) -> None:
        # TODO - Docstring

        # TODO - Refactor

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

                # print(classification['word'])

                if len(set(current_ner_tag)) == 1:
                    current_ner_tag = [current_ner_tag[0]]

                # current_ner_tag = [current_ner_tag[0]]

                if (
                    index < len(classifications) - 1 and
                    not classifications[index + 1]['word'].startswith(
                        tuple(['Ġ', ','])
                    )
                ):
                    continue

                ner_tags.extend(current_ner_tag) if len(
                    current_ner_tag) == 1 else ner_tags.append(current_ner_tag)

                current_ner_tag = self._fix_ner_tags(
                    ner_tags=current_ner_tag,
                    row=i,
                    word_index=word_index
                )

                fixed_ner_tags.extend(current_ner_tag) if len(
                    current_ner_tag) == 1 else fixed_ner_tags.append(current_ner_tag)

                word_index += 1

                current_ner_tag = []

            ner_tags_col.append(ner_tags)
            fixed_ner_tags_col.append(fixed_ner_tags)

        self._data['ner_tags'] = ner_tags_col
        self._data['fixed_ner_tags'] = fixed_ner_tags_col

        self._save_data()
