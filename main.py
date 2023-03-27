import os

from text_parsing_methods_using_nlp.annotation.annotator import Annotator
from text_parsing_methods_using_nlp.models.slovakbert_ner_model import (
    SlovakBertNerModel
)

if __name__ == '__main__':

    # # Annotation
    # for i in range(100, 451, 50):
    #     annotator = Annotator(dataset_size=i)

    #     annotator()

    #     del annotator

    # Model
    model = SlovakBertNerModel()

    model()
