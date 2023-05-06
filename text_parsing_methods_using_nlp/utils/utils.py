import json
import os
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import spacy
from spacy import displacy
from spacy.tokens import Doc

from text_parsing_methods_using_nlp.config.config import (
    INVERTED_NER_LABELS,
    NER_ENTITY_COLORS
)
from text_parsing_methods_using_nlp.config.wandb_config import (
    WAND_ENV_VARIABLES
)

# region Model Setup


def setup_weights_and_biases_env_variables() -> None:
    """Sets up Weights and Biases Environmental Variables."""
    for variable, value in WAND_ENV_VARIABLES.items():
        if variable not in os.environ or os.getenv(variable) != value:
            os.environ[variable] = value

# endregion

# region Training History Processing


def extract_train_and_eval_evaluations(
    training_history_path: str
) -> List[Dict[str, Union[int, float]]]:
    """Extracts Training History Train and Eval Data for each step.

    Args:
        `training_history_path` (str): Training History file path.

    Returns:
        List[Dict[str, Union[int, float]]]: Extracted Train and Eval Data.
    """
    train_and_eval_evaluations = []

    with open(training_history_path) as json_file:
        evaluations = json.load(json_file)['log_history']

    for evaluation in evaluations:
        if any(
            [
                key in evaluation.keys() for key in
                ['loss', 'eval_loss']
            ]
        ):
            train_and_eval_evaluations.append(evaluation)

    return train_and_eval_evaluations


def extract_step_history(
    train_and_eval_evaluations: List[Dict[str, Union[int, float]]]
) -> List[Dict[str, Union[int, float]]]:
    """Merges Training History Train and Eval Data for each step.

    Args:
        `train_and_eval_evaluations` (List[Dict[str, Union[int, float]]]): 
        Training History Steps Data.

    Returns:
        List[Dict[str, Union[int, float]]]: Merged Training History Data.
    """
    step_history = []

    for i in range(0, len(train_and_eval_evaluations) - 1, 2):
        step_history.append(
            {
                **train_and_eval_evaluations[i],
                **train_and_eval_evaluations[i+1]
            }
        )

    return step_history


def process_training_history(training_history_path: str) -> pd.DataFrame:
    """Processes training history file.

    Args:
        `training_history_path` (str): Training History file path.

    Returns:
        pd.DataFrame: Processed Training History.
    """
    return pd.DataFrame(
        extract_step_history(
            extract_train_and_eval_evaluations(training_history_path)
        )
    )

# endregion

# region Prediction Visualization


def process_entity_ranges(
    classifications: Dict[str, Any]
) -> List[Tuple[str, int, int]]:
    """Processes Entity ranges.

    Args:
        `classifications` (Dict[str, Any]): Model generated classifications.

    Returns:
        List[Tuple[str, int, int]]: Entity ranges.
    """
    entities = []

    for i in range(len(classifications)):
        if classifications[i]['entity'] != '0':
            if classifications[i]['entity'][0] == 'B':
                j = i + 1
                while (
                    j < len(classifications) and
                    classifications[i]['entity'].split(
                        '-')[-1] == classifications[j]['entity'].split('-')[-1]
                ):
                    j += 1
                entities.append(
                    (
                        classifications[i]['entity'].split('-')[-1],
                        classifications[i]['start'],
                        classifications[j - 1]['end']
                    )
                )

    return entities


def apply_ner_labels(
    input_sentence: str,
    entities: List[Tuple[str, int, int]]
) -> Doc:
    """Label document based on entities.

    Args:
        `input_sentence` (str): Input sentence to annotate.

        `entities` (List[Tuple[str, int, int]]): Entity ranges in 
        input_sentence.

    Returns:
        Doc: Labelled document.
    """
    nlp = spacy.blank('sk')

    doc = nlp(input_sentence)

    ents = []
    for ee in entities:
        word = doc.char_span(ee[1], ee[2], ee[0])

        if not ents:
            previous_range = range(-1, -1)

        current_range = range(ee[1], ee[2])

        if word and not (
            current_range.start in previous_range and
            current_range[-1] in previous_range
        ):
            ents.append(word)

        previous_range = current_range

    doc.ents = ents

    return doc


def display_ner_classification(
    input_sentence: str,
    classifications: Dict[str, Any]
) -> None:
    """Displays NER Classification Visualization.

    Args:
        `input_sentence` (str): Input sentence to display.

        `classifications` (Dict[str, Any]): Model generated classifications.
    """
    displacy.serve(
        docs=apply_ner_labels(
            input_sentence=input_sentence,
            entities=process_entity_ranges(classifications)
        ),
        style='ent',
        options={
            'ents': NER_ENTITY_COLORS.keys(),
            'colors': NER_ENTITY_COLORS
        },
        auto_select_port=True
    )

# endregion

# region Utils


def setup_folders(folders: List[str]) -> None:
    """Creates missing directories.

    Args:
        `folders` (List[str]): List of missing directories.
    """
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)

# endregion
