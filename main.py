import os

import text_parsing_methods_using_nlp as tpm_nlp


def run_annotation() -> None:
    for i in range(100, 451, 50):
        annotator = tpm_nlp.Annotator(dataset_size=i)

        annotator()

        del annotator


def run_modeling() -> None:
    for v in ['0.0.4']:
        model = tpm_nlp.SlovakBertNerModel(version=v)

        model()

        del model


if __name__ == '__main__':
    run_modeling()
