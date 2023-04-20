import os
import json
from typing import Dict, List, Union

import pandas as pd

from text_parsing_methods_using_nlp.wandb_config import WAND_ENV_VARIABLES


def setup_weights_and_biases_env_variables() -> None:
    # TODO - Docstring

    for variable, value in WAND_ENV_VARIABLES.items():
        if variable not in os.environ or os.getenv(variable) != value:
            os.environ[variable] = value

def setup_folders(folders: List[str]) -> None:
    # TODO - Docstring

    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)

def extract_train_and_eval_evaluations(
    training_history_path: str
) -> List[Dict[str, Union[int, float]]]:
    # TODO - Docstring

    train_and_eval_evaluations = []

    with open(training_history_path) as json_file:
        evaluations = json.load(json_file)['log_history']

    for evaluation in evaluations:            
        if any(
            [
                key in evaluation.keys() for key in 
                ['train_accuracy', 'eval_accuracy']
            ]
        ):
            train_and_eval_evaluations.append(evaluation)

    return train_and_eval_evaluations


def extract_step_history(
    train_and_eval_evaluations: List[Dict[str, Union[int, float]]]
) -> List[Dict[str, Union[int, float]]]:
    # TODO - Docstring

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
    # TODO - Docstring

    return pd.DataFrame(
        extract_step_history(
            extract_train_and_eval_evaluations(training_history_path)
        )
    )
