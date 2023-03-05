import os

from text_parsing_methods_using_nlp.annotation.annotator import Annotator


if __name__ == '__main__':
    dataset_size = 100

    annotator = Annotator(
        dataset_size=dataset_size,
        manual_correction_filepath=os.path.join(
            os.getcwd(),
            'text_parsing_methods_using_nlp',
            'data',
            'manual_correction_files',
            f'manual_{dataset_size}.txt'
        )
    )

    annotator()
