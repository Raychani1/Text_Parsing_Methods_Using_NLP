import itertools
import os
from typing import Any, Dict, List

import pandas as pd
from natsort import natsorted
from transformers import (
    pipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

import text_parsing_methods_using_nlp as tpm_nlp
from text_parsing_methods_using_nlp.config.config import (
    MODEL_OUTPUT_FOLDER,
    MODEL_TEST_DATASET_FOLDER,
)
from text_parsing_methods_using_nlp.config.model_config import (
    MODEL_CONFIG_V_POC,
    MODEL_CONFIG_V0_0_1,
    MODEL_CONFIG_V0_0_2,
    MODEL_CONFIG_V0_0_3,
    MODEL_CONFIG_V0_0_4,
    MODEL_CONFIG_V0_0_5,
    MODEL_CONFIG_V0_0_6
)


def run_annotation(model_name: str, model_path: str) -> None:
    """Executes annotation process with specific model.

    Args:
        `model_name` (str): Model name to use in annotation process evaluation 
        plots.
        `model_path` (str): Path to model to use in annotation process.
    """
    annotator = tpm_nlp.Annotator(
        model=model_path,
        model_name=model_name,
        tokenizer=model_path,
    )

    annotator()


# region Modeling


def run_model_training(config: Dict[str, Any]) -> None:
    """Executes model training based on configurations.

    Args:
        `config` (Dict[str, Any]): Model configurations to be used for
        training.
    """
    model = tpm_nlp.SlovakBertNerModel(**config)
    model()

    del model


def run_early_stopping_training(
    config: Dict[str, Any],
    patience_values: List[int]
) -> None:
    """Executes Early Stopping Training.

    Args:
        `config` (Dict[str, Any]): Model configurations to be used for
        training.
        `patience_values` (List[int]): Patience values for training.
    """
    for patience in patience_values:
        config['early_stopping_patience'] = patience

        run_model_training(config)


def run_hyperparam_tuning_training(
    config: Dict[str, Any],
    number_of_iterations: int
) -> None:
    """Executes Hyperparameter Tuning Training.

    Args:
        `config` (Dict[str, Any]): Model configurations to be used for
        training.
        `number_of_iterations` (int): Number of iterations to repeat training.
    """
    for _ in range(number_of_iterations):
        run_model_training(config)


def run_dropout_training(
    config: Dict[str, Any],
    dropout_values: List[int]
) -> None:
    """Executes Dropout Value Modification Training.

    Args:
        `config` (Dict[str, Any]): Model configurations to be used for
        training.
        `dropout_values` (List[int]): Dropout values for training.
    """
    for (
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        classifier_dropout_value
    ) in [
        p for p in itertools.product(
            dropout_values,
            repeat=3
        )
    ]:
        config['hidden_dropout_prob'] = hidden_dropout_prob
        config['attention_probs_dropout_prob'] = (
            attention_probs_dropout_prob
        )
        config['classifier_dropout_value'] = (
            classifier_dropout_value
        )

        run_model_training(config)


def run_freezing_training(config: Dict[str, Any]) -> None:
    """Executes Layer Freezing Training.

    Args:
        `config` (Dict[str, Any]): Model configurations to be used for
        training.
    """
    for freezing in [
        [],
        range(0, 1),
        range(0, 3),
        range(0, 5),
        range(0, 7),
        range(0, 10),
        range(0, 12),
    ]:
        layers_to_freeze = ['embeddings']
        layers_to_freeze.extend(freezing)

        config['layers_to_freeze'] = layers_to_freeze

        run_model_training(config)


def run_modeling() -> None:
    """Executes Every Modeling Training."""
    for config in [
        MODEL_CONFIG_V_POC,
        MODEL_CONFIG_V0_0_1,
        MODEL_CONFIG_V0_0_2,
        MODEL_CONFIG_V0_0_3,
        MODEL_CONFIG_V0_0_4,
        MODEL_CONFIG_V0_0_5,
        MODEL_CONFIG_V0_0_6
    ]:
        if config['version'] == '0.0.3':
            run_early_stopping_training(config=config, patience_values=[3, 10])

        elif config['version'] == '0.0.4':
            run_hyperparam_tuning_training(
                config=config,
                number_of_iterations=10
            )

        elif config['version'] == '0.0.5':
            run_dropout_training(
                config=config,
                dropout_values=[0.15, 0.2, 0.25]
            )

        elif config['version'] == '0.0.6':
            run_freezing_training(config=config)

        else:
            run_model_training(config=config)

# endregion


def run_evaluation(versions: List[str]) -> None:
    """Runs evaluation for every passed model version.

    Args:
        `versions` (List[str]): Model versions to evaluate.
    """
    for version in versions:
        version_folder = [
            s for s in os.listdir(MODEL_OUTPUT_FOLDER) if version in s
        ][0]

        model_version_folder = os.path.join(
            MODEL_OUTPUT_FOLDER,
            version_folder
        )

        for model_name in natsorted(os.listdir(model_version_folder)):
            input_file_path = [
                s for s in os.listdir(MODEL_TEST_DATASET_FOLDER)
                if model_name in s
            ]

            if not input_file_path:
                input_file_path = [
                    s for s in os.listdir(MODEL_TEST_DATASET_FOLDER)
                    if model_name[:-2] in s
                ]

            model = tpm_nlp.SlovakBertNerModel(version=version)

            model.evaluate(
                input_data_filepath=os.path.join(
                    MODEL_TEST_DATASET_FOLDER,
                    input_file_path[-1]
                ),
                dataset_size={'_POC': 86, '0.0.1': 7895}.get(version, 6254),
                model_name=model_name,
                model_input_folder=os.path.join(
                    model_version_folder,
                    model_name
                ),
                plot_history=True
            )

            del model


def run_prediction(model_path: str) -> None:
    """Executes prediction with a specific model.

    Args:
        `model_path` (str): Path to model used for prediction.
    """
    input_sentence = input('Please provide Input to predict: ')

    ner_pipeline = pipeline(
        'ner',
        model=AutoModelForTokenClassification.from_pretrained(model_path),
        tokenizer=AutoTokenizer.from_pretrained(model_path)
    )

    classifications = ner_pipeline(input_sentence)

    tpm_nlp.display_ner_classification(
        input_sentence=input_sentence,
        classifications=classifications
    )


if __name__ == '__main__':
    run_annotation(model_name='my-model', model_path='path/to/my-model')

    run_modeling()

    run_evaluation(
        versions=[
            '_POC', '0.0.1', '0.0.2', '0.0.3', '0.0.4', '0.0.5', '0.0.6'
        ]
    )

    run_prediction(model_path='path/to/my-model')
