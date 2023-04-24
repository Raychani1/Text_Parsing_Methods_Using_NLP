import itertools
import os

import spacy
from spacy import displacy

from natsort import natsorted
from transformers import (
    pipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
    IntervalStrategy
)

import text_parsing_methods_using_nlp as tpm_nlp
from text_parsing_methods_using_nlp.config import (
    MODEL_OUTPUT_FOLDER,
    MODEL_TEST_DATASET_FOLDER,
    INVERTED_NER_LABELS
)

from pprint import pprint

# TODO - Previous version configurations

MODEL_CONFIG_V_POC = {
    'version': '_POC',
    'hyperparameter_tuning':  False,
    'concat_with_wikiann': False,
    'do_train': True,
    'do_eval': True,
    'do_predict': True,
    'evaluation_strategy': IntervalStrategy.EPOCH,
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 4,
    'num_train_epochs': 15,
    'logging_strategy': IntervalStrategy.EPOCH,
    'save_strategy': IntervalStrategy.EPOCH,
    'save_total_limit': 2,
    'eval_steps': 500,
    'report_to': 'wandb',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
}


MODEL_CONFIG_V0_0_5 = {
    'version': '0.0.5',
    'early_stopping_patience': 3,
    'hyperparameter_tuning':  False,
    'do_train': True,
    'do_eval': True,
    'do_predict': True,
    'evaluation_strategy': IntervalStrategy.EPOCH,
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 4,
    'learning_rate': 0.000024981604754,
    'weight_decay': 0.285214291922975,
    'num_train_epochs': 15,
    'logging_strategy': IntervalStrategy.EPOCH,
    'save_strategy': IntervalStrategy.EPOCH,
    'save_total_limit': 2,
    'eval_steps': 500,
    'report_to': 'wandb',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
}


def run_annotation() -> None:
    for i in range(100, 451, 50):
        annotator = tpm_nlp.Annotator(dataset_size=i)

        annotator()

        del annotator


def run_modeling() -> None:
    # V0.0.5
    # for (
    #     hidden_dropout_prob,
    #     attention_probs_dropout_prob,
    #     classifier_dropout_value
    # ) in [
    #     p for p in itertools.product(
    #         [0.15, 0.2, 0.25],
    #         repeat=3
    #     )
    # ]:
    #     MODEL_CONFIG_V0_0_5['hidden_dropout_prob'] = hidden_dropout_prob
    #     MODEL_CONFIG_V0_0_5['attention_probs_dropout_prob'] = (
    #         attention_probs_dropout_prob
    #     )
    #     MODEL_CONFIG_V0_0_5['classifier_dropout_value'] = (
    #         classifier_dropout_value
    #     )

    #     model = tpm_nlp.SlovakBertNerModel(**MODEL_CONFIG_V0_0_5)
    #     model()

    #     del model

    # V POC
    model = tpm_nlp.SlovakBertNerModel(**MODEL_CONFIG_V_POC)
    model()


def run_evaluation() -> None:
    # TODO - Docstring

    for version in ['_POC', '0.0.1', '0.0.2', '0.0.3', '0.0.4', '0.0.5']:
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
                )
            )

            del model


def run_prediction(model_path):
    # TODO - Docstring

    ner_pipeline = pipeline(
        'ner',
        model=AutoModelForTokenClassification.from_pretrained(model_path),
        tokenizer=AutoTokenizer.from_pretrained(model_path)
    )

    input_sentence = input('Please provide Input to predict: ')

    classifications = ner_pipeline(input_sentence)

    entities = []
    for i in range(len(classifications)):
        if classifications[i]['entity'] != '0':
            print(classifications[i]['entity'])

            if classifications[i]['entity'][0] == 'B':
                j = i + 1
                while j < len(classifications) and classifications[i]['entity'].split('-')[-1] == classifications[j]['entity'].split('-')[-1]:
                    j += 1
                entities.append(
                    (
                        classifications[i]['entity'].split('-')[-1], 
                        classifications[i]['start'],
                        classifications[j - 1]['end']
                    )
                )

    nlp = spacy.blank("sk")  # it should work with any language

    doc = nlp(input_sentence)

    ents = []
    for ee in entities:
        print(ee)

        word = doc.char_span(ee[1], ee[2], ee[0])

        if not ents:
            previous_range = range(-1, -1)

        current_range = range(ee[1], ee[2])
              

        if word and not (
            current_range.start in previous_range and current_range[-1] in previous_range
        ):
            ents.append(word)

        previous_range = current_range

    doc.ents = ents

    options = {
        "ents": [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            'DATE',
            'TIME',
            'MONEY',
            'PERCENTAGE'
        ],
        "colors": {
            "PERSON": "lightblue",
            "ORGANIZATION": "lightcoral",
            "LOCATION": "#FFB703",
            'DATE': "#EF233C",
            'TIME': "#CCFF33",
            'MONEY': "lightgreen",
            'PERCENTAGE': "#8D99AE"
        }
    }
    displacy.serve(doc, style="ent", options=options, auto_select_port=True)


if __name__ == '__main__':
    # run_modeling()
    # run_evaluation()

    run_prediction(
        model_path='MODEL_PATH'
    )
