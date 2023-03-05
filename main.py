import os

from text_parsing_methods_using_nlp.annotation.annotator import Annotator


if __name__ == '__main__':
    annotator = Annotator(
        dataset_size=100,
        manual_correction_filepath=os.path.join(os.getcwd(), 'manual.txt')
    )

    annotator()
