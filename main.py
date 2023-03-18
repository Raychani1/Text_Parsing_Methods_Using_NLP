import os

from text_parsing_methods_using_nlp.annotation.annotator import Annotator


if __name__ == '__main__':

    for i in range(100, 451, 50):
        annotator = Annotator(dataset_size=i)

        annotator()

        del annotator
