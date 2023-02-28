import csv
import os
import re
from typing import List

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

    def _preprocess_data(self) -> None:
        # TODO - Docstring

        regex_mapping = {

            # Add an extra space before colon which is not a time indicator
            r'([a-zA-Z]+):([a-zA-Z]+)': r'\1 : \2',

            r'(\[\s*[0-9]+\s*\])': r' ',

            # Remove special characters
            r'''['"`‘’„“”\(\)\[\]\/(\s\-|–\s))!?;]|(\s*:\s+)|(\.|,)\s*(\.|,)''': r' ',

            # Remove dots from text
            r'([a-zA-Z]{2,})(\s*\.\s*)': r'\1 ',

            # Remove dots from text
            r'([ľščťžýáíéóúäôňďĺ%]{1,}[a-zA-Z0-9]*)\s*(\.)\s*': r'\1 ',

            # Remove dots from text
            r'([0-9]{1,})(\s*\.\s*)': r'\1 ',

            # Replace floating point number commas with dot notation
            r'([+-]?[0-9]+),([0-9]+)': r'\1.\2',

            # Add extra space to floating point percentages
            r'([+-]?[0-9]+\.[0-9]+|[+-]?[0-9])(%)': r'\1 \2 ',

            r',': r' ',

            # # Remove dots from text
            # r'([a-zA-Z]{2,})(\.\s*)([a-zA-Z]*)': r'\1 \3',

            # Replace excessive whitespace characters
            r'\s+': r' ',

            # Merge larger number formats together
            r'( [+-]?[0-9]{1,3}) ([0-9]+) ([a-z]*)': r'\1\2 \3',

            # r'j\.\s*s\.\s*a': 'j.s.a',

            # r'a\.\s*s\.': 'a.s.',

            # r'atď .': 'atď.',

            r'€': 'euro',

            r'(\s+[a-zA-Zľščťžýáíéóúäôňďĺ]) \.': r'\1.',
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

    def _fix_ner_tags(self, ner_tags: List[int], row: int, word_index: int):
        # TODO - Docstring

        new_ner_tags = []

        text_tokens = self._data['text'][row].split()
        lemma_text_tokens = self._data['lemma_text'][row].split()
        
        current_word = text_tokens[word_index]
        current_word_lemma = lemma_text_tokens[word_index]

        print(current_word)


        next_word = next_word_lemma = ''

        if word_index < len(text_tokens) - 1:
            next_word = text_tokens[word_index + 1]
            next_word_lemma = lemma_text_tokens[word_index + 1]

        if current_word in ['NBS', 'NAKA'] and ner_tags[0] not in [3, 4]:
            new_ner_tags.append(3)
        elif re.compile(r'[0-9]{2}:[0-9]{2}').match(current_word):
            new_ner_tags.append(9)
        elif re.compile(r'[+-]?[0-9]+\.[0-9]+|[+-]?[0-9]').match(current_word):
            if next_word == '%':
                new_ner_tags.append(12)
            elif next_word_lemma in ['mil', 'miliarda', 'milión']:
                new_ner_tags.append(10)
        elif current_word_lemma in ['%', 'percento']:
            new_ner_tags.append(13)
        else:
            new_ner_tags = ner_tags

        return new_ner_tags

    def annotate(self) -> None:
        # TODO - Docstring

        # TODO - Refactor

        self._preprocess_data()

        ner_tags_col = []

        for i in range(self._dataset_size):
            classifications = self._ner_pipeline(self._data['text'][i])

            word_index = 0
            ner_tags = []
            current_ner_tag = []

            for index, classification in enumerate(classifications):
                current_ner_tag.append(classification['entity'])

                print(classification['word'])

                if len(set(current_ner_tag)) == 1:
                    current_ner_tag = [current_ner_tag[0]]

                if (
                    index < len(classifications) - 1 and
                    not classifications[index + 1]['word'].startswith(
                        tuple(['Ġ', ','])
                    )
                ):
                    continue

                current_ner_tag = self._fix_ner_tags(
                    ner_tags=current_ner_tag,
                    row=i,
                    word_index=word_index
                )

                ner_tags.extend(current_ner_tag) if len(
                    current_ner_tag) == 1 else ner_tags.append(current_ner_tag)

                word_index += 1

                current_ner_tag = []

            ner_tags_col.append(ner_tags)

        self._data['ner_tags'] = ner_tags_col

        self._save_data()
