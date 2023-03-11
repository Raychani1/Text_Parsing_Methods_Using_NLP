import os

from text_parsing_methods_using_nlp.annotation.annotator import Annotator


if __name__ == '__main__':
    dataset_size = 100

    annotator = Annotator(dataset_size=dataset_size)

    annotator()
